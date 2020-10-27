#!/usr/bin/env python
# coding: utf-8

# QMix applied to pommerman

# In[2]:


import random
from collections import namedtuple
import numpy as np
from copy import deepcopy


# In[3]:


import pommerman
from pommerman import agents
from pommerman import constants


# In[4]:


import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


# In[5]:


from IPython.core.debugger import set_trace


# In[6]:


def featurize(obs):
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


# In[7]:


Transition = namedtuple('Transition',
                        ('state', 'actions', 'next_state', 'rewards', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def push_episode(self, episode):
        for transition in episode:
            self.push(*transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# In[8]:


def generate_episode(env, render=False):
    state, done = env.reset(), False
    episode = []
    while not done:
        if render:
            env.render()
        actions = env.act(state)
        next_state, rewards, done, info = env.step(actions)
        episode.append(Transition(state, actions, next_state, rewards, done))
        state = next_state
    return episode


# In[9]:


def episode_stats(env, episode):
    """Collects statistics from episode
    For now, only about actions taken"""
    
    from collections import Counter
    action_counter = {}
    for idx in range(len(env._agents)):
        action_counter[idx] = Counter()
    for transition in episode:
        for idx, action in enumerate(transition.actions):
            action_counter[idx][list(constants.Action)[action].name] += 1
    return {'actions': action_counter}


# https://github.com/starry-sky6688/StarCraft

# In[10]:


class QMixAgent(agents.BaseAgent):
    def __init__(self, agent_idx):
        super().__init__()
        self.index = agent_idx
        self.epsilon = 1.0
        self.model  = QMixModel().to(args.device)
        self.target = QMixModel().to(args.device)
        self.sync_models()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr)
        self.mode = 'train' # 'train' or 'eval'
    
    def __repr__(self):
        return f'QMix{self.index}'
    
    __str__ = __repr__
    
    def set_mode(self, mode):
        assert mode in ['train', 'eval'], f"Mode {mode} not allowed"
        self.mode = mode
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
        
    def update(self, batch):
        self.mode = 'train'
        states, actions, next_states, rewards, dones = list(zip(*batch))
        obs      = [state[self.index] for state in states]
        actions  = [action[self.index] for action in actions]
        next_obs = [state[self.index] for state in next_states]
        rewards  = torch.tensor([reward[self.index] for reward in rewards]).float().to(args.device)
        dones    = torch.tensor(dones).float().to(args.device)
        
        q_vals  = self.model([featurize(o) for o in obs])
        qa_vals = q_vals[range(len(obs)), actions]
        
        q_vals  = self.target([featurize(o) for o in next_obs])
        q_vals_max, _ = torch.max(q_vals, dim=1)
        td_error = rewards + (1-dones) * args.gamma * q_vals_max - qa_vals
        
        self.model.zero_grad()
        loss = torch.mean(td_error**2)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _update_epsilon(self):
        self.epsilon *= args.eps_decay 
        self.epsilon = max(args.eps_min, self.epsilon)
        
    def act(self, obs, action_space):
        if self.mode == 'train':
            self._update_epsilon()
            if np.random.random() < self.epsilon:
                return np.random.choice(constants.Action).value
            else:
                Qs = self.model([featurize(obs)])
                return torch.argmax(Qs).item()
        elif self.mode == 'eval':
            Qs = self.model([featurize(obs)])
            return torch.argmax(Qs).item()
        else:
            raise ValueError(f'Invalid mode: {self.mode}')


# In[11]:


class QMixModel(nn.Module):
    def __init__(self, h=11, w=11, c=18, outputs=len(constants.Action)):
        super().__init__()
        # input is batch of tensors of size 11x11x18
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=64,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, stride=1)
        
        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) -1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 128
        
        self.fc  = nn.Linear(linear_input_size, linear_input_size)
        self.out = nn.Linear(linear_input_size, outputs)
    
    def forward(self, obs):
        if isinstance(obs, list):
            x = torch.from_numpy(np.array(obs)).float().to(args.device)
        else:
            x = torch.from_numpy(obs).float().to(args.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.out(x).to(args.device)


# In[12]:


def train(env, training_agents, n_steps=10000): 
    buffer = ReplayMemory(capacity=args.buffer_size)
    state, done = env.reset(), False
    running_loss = None
    for step_idx in range(n_steps):
        actions = env.act(state)
        next_state, rewards, done, info = env.step(actions)
        buffer.push(state, actions, next_state, rewards, done)
        if len(buffer) < args.min_train_size:
            continue
        batch = buffer.sample(args.batch_size)
        for agent in training_agents:
            loss = agent.update(batch)
            running_loss = loss if running_loss is None else args.alpha*running_loss + (1-args.alpha)*loss
            if args.verbose and step_idx % args.print_interval == 0:
                print(f"Step {step_idx:3d} - Agent {agent}: loss = {running_loss:.5f}")
    return


# In[13]:


def eval(env, training_agents, n_episodes=10):
    "Generates n_episodes episodes and returns average final reward for all training agents"
    for agent in training_agents:
        agent.set_mode('eval')
    rewards = [generate_episode(env)[-1].rewards for _ in range(n_episodes)]
    agent_reward = {}
    for agent in training_agents:
        agent_reward[agent] = sum([reward[agent.index] for reward in rewards])/n_episodes
    return agent_reward


# In[14]:


def runner():
    training_agents = [QMixAgent(0)]
    other_agents    = [agents.SimpleAgent()]
    agent_list = training_agents + other_agents
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    for epoch in range(args.n_epochs):
        train(env, training_agents, n_steps=args.n_steps)
        rewards = eval(env, training_agents)
        print(f"Epoch {epoch:3d} - Wins = {rewards}")
        stats   = episode_stats(env, generate_episode(env, render=False))
        for agent in training_agents:
            print('\t', agent, ' : ', stats['actions'][agent.index])


# In[15]:


class args:
    buffer_size    = int(10e4)
    min_train_size = 100
    batch_size     = 64
    gamma          = 0.9
    lr             = 0.001
    alpha          = 0.9
    print_interval = 20
    n_epochs       = 50
    n_steps        = 200
    verbose        = False
    eps_decay      = 1 - 10e-9
    eps_min        = 0.05
    device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')


# In[16]:


# runner()


# In[35]:


class QMix:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents  = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape  = self.obs_shape
        #input_shape += self.n_actions # add last action to agent network input
        #input_shape += self.n_agents  # reuse agent network for all agents (-> weight sharing)
        
        self.eval_rnn = RNN(input_shape, args)   # the agent network that produces Q_a(.)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)       # the mixer netwok Qtot = f(Q1, ..., Qn, state)
        self.target_qmix_net = QMixNet(args)
        
        # copy weigths from eval to target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        
        self.eval_parameters = list(self.eval_qmix_net.parameters()) +                                list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        
        self.eval_hidden = None
        self.target_hidden = None
        print('Initialized QMix')
    
    def learn(self, max_episode_len, train_step, epsilon=None):
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
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        
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
        torch.nn.utils.clip_grad_norm(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
    
    def _get_inputs(self, batch, transition_idx):
        """Get all obs, next_obs and actions (as onehot vector) at
        transition_idx (over all episodes in batch)"""
        obs, obs_next, u_onehot = batch['o'][:, transition_idx],                                   batch['o_next'][:, transition_idx],                                   batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if self.args.last_action:   # whether to use the last action to choose action
            if transition_idx == 0:  # for first experience, previous action is zero vector
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network: # weight sharing: whether to use one network for all agents
            pass
        
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


# In[37]:


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


# In[38]:


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        
        # hyper_w1 has to be a matrix, and pytorch nn.Linear only outputs a vector,
        # first output vector of length n_row*n_cols and convert it to matrix
        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden)
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


# In[33]:


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents  = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMix(args)
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
            if key != 'z': # TODO: what is 'z'? -> MAVEN
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated_shape[0] # number of episode in batch
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len
        


# In[19]:


def featurize_state(env):
    """Returns a tensor of size n_agentsx11x11x18"""
    outs = []
    for obs in env.get_observations():
        outs.append(featurize(obs))
        
    # convert board items into bitmaps
    maps = [board == i for i in range(10)] # returns list of 10 arrays, for each type 0..9

    outs = np.stack(outs, axis=0)
    return outs


# In[20]:


training_agents = [QMixAgent(0)]
other_agents    = [agents.SimpleAgent()]
agent_list = training_agents + other_agents
env = pommerman.make('PommeFFACompetition-v0', agent_list)


# In[21]:


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents   = agents
        self.n_agents = len(agents)
        
        self.args = args
        self.epsilon = args.epsilon
        self.episode_limit = args.episode_limit
        
        print('Init RolloutWorker')
        
    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # padded = list of zeros and ones; one if corresponding values in other lest are 'padded', 
        #  i.e. added to have episode_limit and uniform output size
        
        self.env.reset()
        
        terminated = False
        step = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        
        epsilon = 0 if evaluate else self.epsilon
        
        
        while not terminated and step < self.episode_limit:
            obs = self.env.get_observations()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            
            # store all results 
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # ? epsilon decay update ?
            #if self.args.epsilon_anneal_scale == 'step':
            #    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        # last obs
        o.append(obs)
        s.append(state)
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
            episode[key] = np.array([episode[key]])
        
        return episode, episode_reward, win_tag


# In[22]:


class Environment: # wrapper for pommerman environment
    def __init__(self, type='PommeFFACompetition-v0'):
        self.agents = [QMixAgent(0)]
        self.n_agents = len(self.agents)
        self.opponents   = [agents.SimpleAgent()]
        agent_list = self.agents + self.opponents
        self.env = pommerman.make(type, agent_list)
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
    
    def get_agents(self):
        return self.agents
    
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
            action = self.opponent.act(observations[self.n_agents+idx], self.env.action_space)
            actions += action
        _, reward, done, _ = env.step(actions)
        info = {}
        if done:
            if reward == 1:
                info = {'battle_won': True}
            elif reward == -1:
                info = {'battle_won': False}
        return reward, done, info
        
    
    def close(self):
        self.env.close()


# In[27]:


class args:
    buffer_size    = int(10e4)
    min_train_size = 100
    batch_size     = 64
    gamma          = 0.9
    lr             = 0.001
    alpha          = 0.9
    print_interval = 20
    n_epochs       = 50
    n_steps        = 200
    verbose        = False
    epsilon        = 1.0
    eps_decay      = 1 - 10e-9
    eps_min        = 0.05
    episode_limit  = 100
    n_actions      = len(pommerman.constants.Action)
    state_shape    = (1, 18, 11, 11)
    obs_shape      = (18, 11, 11)
    device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')


# In[36]:


env = Environment()
args.n_agents = env.n_agents
row = RolloutWorker(env, Agents(args), args)
row.generate_episode()


# In[18]:


class Runner:
    def __init__(self, env, args):
        self.args = args
        self.env = pommerman.make('PommeFFACompetition-v0', agent_list)
        self.agents = Agents(env)
        
        self.rollout_worker = RolloutWorker()
        
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
            
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, _, _ = self.rollout_worker.generate_episode(episode_idx)
                episodes.append(episode)
            episode_batch = episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys(): # TODO: what does this do?
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]),
                                                        axis=0)
                self.buffer.store_episode(episode_batch)
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
        return (win_number / self.args.evaluate_epochs,                episode_rewards / self.args.evaluate_epochs)
            


# In[19]:


runner = Runner()
runner.run(0)

