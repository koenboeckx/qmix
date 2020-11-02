#!/usr/bin/env python
# coding: utf-8

# QMix applied to pommerman

import random
import os
from collections import namedtuple
import numpy as np
from copy import deepcopy

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent
from pommerman import constants


import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from replay_buffer import MyReplayBuffer as ReplayBuffer


class QMix: # policy
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents  = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.model_dir = args.model_dir
        input_shape  = self.obs_shape
        input_shape += self.n_actions # add last action to agent network input
        input_shape += self.n_agents  # reuse agent network for all agents (-> weight sharing)
        
        self.eval_rnn = RNN(input_shape, args)   # the agent network that produces Q_a(.)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)       # the mixer network Qtot = f(Q1, ..., Qn, state)
        self.target_qmix_net = QMixNet(args)
        
        # copy weigths from eval to target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        
        self.eval_parameters = list(self.eval_qmix_net.parameters()) +                                list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        
        self.eval_hidden = None
        self.target_hidden = None
        print('Initialized QMix')
    
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        """
        In learning, the extracted data is four-dimensional, and the four dimensions
        are:
            1-which is the first episode
            2-which is the transition of the episode
            3—The data of which agent 
            4—Specific obs dimension. 
        Because when selecting an action, not only the current inputs need to be input,
        but also hidden_state is input to the neural network.
        hidden_state is related to previous experience, so you cannot randomly select
        experience for learning. So here we extract multiple episodes at once, and then
        give them to the neural network at a time
        Transition in the same position of each episode
        """
        episode_num = batch['o'].shape[0] # shape of 'o': (number_of_episodes x episode_limit x n_agents x obs_shape)
        self.init_hidden(episode_num)
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'],                                                              batch['u'], batch['r'],                                                              batch['avail_u'], batch['avail_u_next'],                                                              batch['terminated']
        mask = 1 - batch["padded"] # padded = 1 if added zeros to get to episode limit
        
        # 1. Agent networks
        # Get the Q value corresponding to each agent, the dimension is (number of episodes, max_episode_len, n_agents, n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        # select q_vals for u = actions taken + remove unneeded dimension
        q_evals = torch.gather(q_evals, dim=3, index=u.long()).squeeze(3)
        
        # get target_q by maximizing - first, set q[unavail_action] = 0
        q_targets[avail_u_next == 0.0] = -99999
        q_targets = q_targets.max(dim=3)[0]
        
        # 2. QMixer to obtain Qtot
        q_total_eval   = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)
        
        # 3. Compute total loss
        targets  = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error # set td_error to zero for filled-up experience
        
        loss =(masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
    
    def _get_inputs(self, batch, transition_idx):
        """Get all obs, next_obs and actions (as onehot vector) at
        transition_idx (over all episodes in batch)"""
        obs         = batch['o'][:, transition_idx]
        obs_next    = batch['o_next'][:, transition_idx]
        u_onehot    = batch['u_onehot'][:]

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # add last action:
        if transition_idx == 0:  # for first experience, previous action is zero vector
            inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        else:
            inputs.append(u_onehot[:, transition_idx - 1])
        inputs_next.append(u_onehot[:, transition_idx])
        # reuse_network: # weight sharing: whether to use one network for all agents
        inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        
        # transform inputs to episode_num x n_agents x ....
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next
        
        
    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            q_eval, self.eval_hidden     = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            # reshape q_eval back to (max_episode_len, n_agents, n_actions)
            q_eval   = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # The obtained q_eval and q_target are a list, the list contains max_episode_len arrays,
        # the dimension of the array is (number of episodes, n_agents, n_actions)
        # Convert the list into an array of 
        # (number of episodes, max_episode_len, n_agents, n_actions)
        q_evals   = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets
    
    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    
    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
    
    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        
        # hyper_w1 has to be a matrix, and pytorch nn.Linear only outputs a vector,
        # first output vector of length n_row*n_cols and convert it to matrix
        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1))
    
    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents) # (episode_num x max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num x max_episode-len, state_shpe)
        
        w1 = torch.abs(self.hyper_w1(states))                           # (1920, 160)
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim) # (1920, 5, 32)
        
        b1 = self.hyper_b1(states)                      # (1920, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)
        
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)
        
        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_total


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents  = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMix(args)
        self.args = args
        print('Initialized Agents')
    
    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions that can be chosen
                                                          # np.zeros returns the indices of the
                                                          # elements that are non-zero.
        
        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1
        
        inputs = np.hstack((inputs, last_action)) # add last action to RNN inputs
        inputs = np.hstack((inputs, agent_id))    # add agent id to RNN inputs
        
        # pick hidden state corresponding to current agent
        hidden_state = self.policy.eval_hidden[:, agent_num, :] 
        
        # add a first dimension (batchsize=1) to inputs tensor (from (42,) to (1,42))) #TODO check tensor
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(args.device)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0).to(args.device)
        hidden_state = hidden_state.to(args.device)
        
        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        
        # choose action from q value
        q_value[avail_actions == 0.0] = -float('inf')
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)
        else:
            action = torch.argmax(q_value)
        
        return action
    
    def train(self, batch, train_step):
        # different episodes have different lengths, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0] # number of episode in batch
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len
        

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents   = agents
        self.agent_idxs = args.agent_idxs
        self.n_agents = len(self.agent_idxs)        
        
        self.args = args
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        self.n_actions = args.n_actions
        self.epsilon = args.epsilon
        self.episode_limit = args.episode_limit
        
        print('Initialized RolloutWorker')
        
    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # padded = list of zeros and ones; one if corresponding values in other lest are 'padded', 
        #  i.e. added to have episode_limit and uniform output size
        
        self.env.reset()
        
        terminated = False
        step = 0
        episode_reward = 0 # Cumulative reward over episode
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        
        epsilon = 0 if evaluate else self.epsilon
        
        
        while not terminated and step < self.episode_limit:
            obs = self.env.get_observations()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in self.agent_idxs:
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id].flatten(),
                                                   last_action[agent_id], agent_id,
                                                   avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                
                actions.append(action.item())
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            
            # store all results 
            ## TODO: only store for training_agents !!
            o.append(np.stack([o.flatten() for o in obs]))
            #s.append(np.stack([s.flatten() for s in stenv.agents # TODO: check if we don't need env.agents[:]ate]))
            s.append(state.flatten())
            #u.append(np.reshape(actions, [self.n_agents, 1]))
            u.append(np.reshape([actions[idx] for idx in self.agent_idxs],
                                [self.n_agents, 1]))
            
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward # reward for first training agent
            step += 1
            # ? epsilon decay update ?
            #if self.args.epsilon_anneal_scale == 'step':
            #    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        # last obs
        o.append(np.stack([o.flatten() for o in obs]))
        #s.append(np.stack([s.flatten() for s in state]))
        s.append(state.flatten())
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        
        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.]) 
            terminate.append([1.])
        episode = dict(o = o.copy(),
                       s = s.copy(),
                       u = u.copy(),
                       r = r.copy(),
                       avail_u = avail_u.copy(),
                       o_next = o_next.copy(),
                       s_next = s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                      )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array(episode[key])
        
        return episode, episode_reward, win_tag

class QMixAgent:
    def __init__(self, controller):
        assert isinstance(controller, Agents), f'Wrong controller type: {type(controller)}'
        self.controller = controller
    
    def act(self, obs):
        pass



class Environment: # wrapper for pommerman environment
    def __init__(self, env_type, args):
        assert env_type in ['PommeFFACompetition-v0'], f'Environment type {env_type} unknown'     

        self.n_agents = args.n_agents
        self.agents_idxs = args.agent_idxs

        #TODO : make this dynamic (# of opponts depends on game type)
        self.opponents = [SimpleAgent() for _ in range(args.n_opponents)]
        #self.opponents = [RandomAgent() for _ in range(args.n_opponents)]
        agent_list = [BaseAgent() for _ in range(self.n_agents)] + self.opponents
        self.env = pommerman.make(env_type, agent_list)
        self.env.training_agents = None
        self.env.reset()
        self.n_actions = self.env.action_space.n
    
    def _featurize(self, obs):
        """Returns a tensor of size 11x11x18"""
        # TODO: history of n moves?
        board = obs['board']

        # convert board items into bitmaps
        maps = [board == i for i in range(10)]
        maps.append(obs['bomb_blast_strength'])
        maps.append(obs['bomb_life'])

        # duplicate ammo, blast_strength and can_kick over entire map
        maps.append(np.full(board.shape, obs['ammo']))
        maps.append(np.full(board.shape, obs['blast_strength']))
        maps.append(np.full(board.shape, obs['can_kick']))

        # add my position as bitmap
        position = np.zeros(board.shape)
        position[obs['position']] = 1
        maps.append(position)

        # add teammate
        if obs['teammate'] is not None:
            maps.append(board == obs['teammate'].value)
        else:
            maps.append(np.zeros(board.shape))

        # add enemies
        enemies = [board == e.value for e in obs['enemies']]
        maps.append(np.any(enemies, axis=0))

        out = np.stack(maps, axis=2) 
        # transpose to CxHxW
        return out.transpose((2, 0, 1))

    def get_state(self):
        # returns state = np.array of size n_agents x 18x11x11
        return np.stack(self.get_observations())
    
    def reset(self):
        self.env.reset()
    
    def get_observations(self):
        "Returns list of observations for our agents"
        obs = self.env.get_observations()
        return [self._featurize(o) for o in obs[:self.n_agents]]
    
    def get_avail_agent_actions(self, agent_id):
        """return avail_action
        """
        # TODO: iplement this correctly
        return list(range(len(pommerman.constants.Action)))
    
    def step(self, actions):
        """return reward, terminated, info
           info contains 'battle_won' if won
        """
        observations = self.env.get_observations()
        for idx, opponent in enumerate(self.opponents):
            action = opponent.act(observations[self.n_agents+idx], self.env.action_space)
            actions += [action]
        _, reward, done, _ = self.env.step(actions)
        reward = reward[self.agents_idxs[0]] # reward for first agent of team
        info = {}
        if done:
            if  reward == 1: 
                info = {'battle_won': True}
            elif reward == -1:
                info = {'battle_won': False}
        return reward, done, info
        
    def close(self):
        self.env.close()


class Runner:
    def __init__(self, args):
        self.args = args
        env = Environment('PommeFFACompetition-v0', args)
        self.agents = Agents(args)
        
        self.rollout_worker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        
        self.win_rates = []
        self.episode_rewards = []
    
    def run(self, num):
        train_steps = 0
        for epoch in range(self.args.n_epochs):
            print(f"Run {num}, train epoch {epoch}")
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)

            for episode_idx in range(self.args.n_episodes):
                episode, _, _ = self.rollout_worker.generate_episode(episode_idx)
                self.buffer.store_episode(episode)

            for train_step in range(self.args.train_steps):
                # train_steps: to indicate when to sync eval and target models
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
    
    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epochs):
            _, episode_reward, win_tag = self.rollout_worker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return (win_number / self.args.evaluate_epochs, \
                episode_rewards / self.args.evaluate_epochs)

#
class args:
    lr             = 0.01
    gamma          = 0.99
    grad_norm_clip = 0.1 # TODO: find convenient value
    target_update_cycle = 5 # TODO: find convenient value
    save_cycle     = 100 # TODO: find convenient value
    model_dir      = './models/'
    buffer_size    = 100
    batch_size     = 32
    n_epochs       = 2
    evaluate_cycle = 10
    evaluate_epochs = 1
    n_episodes     = 100
    train_steps    = 200
    verbose        = False
    epsilon        = 1.0
    eps_decay      = 1 - 10e-9
    eps_min        = 0.05
    episode_limit  = 77
    n_actions      = len(pommerman.constants.Action)
    n_agents       = 2
    agent_idxs     = range(n_agents)  # list of training agents
    n_opponents    = 2
    obs_shape      = 18 * 11 * 11
    state_shape    = n_agents * obs_shape
    rnn_hidden_dim = 64
    qmix_hidden_dim = 64
    ## TODO: IMPORTANT: correct this:
    # device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device          = torch.device("cpu")
    print(f'Using device {device}')       

# -------------------------------------------------------------------------------
def test_environment(args):
    env = Environment('PommeFFACompetition-v0', args)
    rollout_worker = RolloutWorker(env, Agents(args), args)
    episode, episode_reward, win_tag = rollout_worker.generate_episode()
    for key in episode.keys():
        print(f"{key} : {episode[key].shape}")
    print(episode_reward, win_tag)

if __name__ == '__main__':
    runner = Runner(args)
    for run in range(10):
        runner.run(run)
        win_rate, _ = runner.evaluate()
        print(f'Win rate: {win_rate}')
    


