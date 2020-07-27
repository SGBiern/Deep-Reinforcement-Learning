import numpy as np
import random
import copy
from collections import deque, namedtuple

from model_project import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0.0001


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size, random_seed)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        # Get states, actions, rewards, next_states, and dones from experiences
        states, actions, rewards, next_states, dones = experiences
        
        # Update Critic
        # get next actions by extrapolating from actor_target (Off policy)
        next_actions = self.actor_target(next_states)
        
        # Derive Q_targets
        next_Q_targets = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma*next_Q_targets*(1-dones))
        
        # Q for current states
        Q_expected = self.critic_local(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimum the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        # Update Actor
        actions_pred = self.actor_local(states).to(device)
        actor_loss = - self.critic_local(states, actions_pred).mean()
        
        # Minimum the loss
        
        actor_loss.backward()
        self.actor_optim.step()
        
        # Soft Update
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
    
    def soft_update(self, local, target,tau):
        for param_local, param_target in zip(local.parameters(), target.parameters()):
            param_target.data.copy_(tau*param_local.data + (1.0-tau)*param_target.data)

    def act(self, state, add_noise=True):
        '''Returns actions for given state as per current policy'''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()

class OUNoise:
    '''Ornstein-Uhlenbeck process'''
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu *np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        '''Reset the internal state (=noise) to mean (mu)'''
        self.state = copy.copy(self.mu)
    
    def sample(self):
        '''Update internal state and return it as a noise sample'''
        x= self.state
        dx = self.theta * (self.mu - x) + self.sigma* np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
