import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model_project_ import Actor, Critic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(
        self, 
        state_size=None,        # state space size
        action_size=None,       # action size
        memory=None,
        buffer_size=int(1e6),   # replay buffer size
        batch_size=128,         # minibatch size
        gamma=0.99,             # discount factor
        tau=1e-3,               # for soft update of target parameters
        lr_actor=1e-4,          # learning rate of the actor 
        lr_critic=1e-3,         # learning rate of the critic
        weight_decay=0,         # L2 weight decay
        random_seed=0
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size       # replay buffer size
        self.batch_size = batch_size         # minibatch size
        self.gamma = gamma                   # discount factor
        self.tau = tau                       # for soft update of target parameters
        self.lr_actor = lr_actor             # learning rate of the actor 
        self.lr_critic = lr_critic           # learning rate of the critic
        self.weight_decay = weight_decay     # L2 weight decay
        self.seed = random.seed(random_seed)
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        # Replay memory
        if not isinstance(memory, ReplayBuffer):
            memory = ReplayBuffer(buffer_size, batch_size, random_seed)
        self.memory = memory

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
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

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MADDPG():
    def __init__(self, state_size, action_size, seed=0, n_agents=2, buffer_size=int(1e6), batch_size=128, gamma=0.99,
                update_every=2, noise_start=1.0, noise_decay=1.0, t_stop_noise=int(3e4)):
        '''
        Params
        ======
            action_size (int) : action dimension
            n_agent (int) : # of agents
            seed (int) : random seed
            buffer_size (int) : replay buffer size (to avoid sampling correlated ones, it needs to be enough large)
            batch_size (int) :  minibatch size
            gamma (float) :  discount factor
            noise_start (float) : initial noise weight
            noise_decay (float) : noise decay rate
            update_every (int) : how often to update the network
            t_stop_noise (int) :  max. number of timesteps with noise application during training
        '''
        super(MADDPG, self).__init__()
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        
        self.n_agents = n_agents
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        
        self.t_step = 0
        self.noise_on = True
        self.t_stop_noise = t_stop_noise
        

        models = [model.Actor_Critic_Model(n_agents=n_agents) for _ in range(n_agents)]
        self.agents = [DDPG(i, models[i]) for i in range(n_agents)]
        
        
        
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1,-1)
        all_next_states = all_next_states.reshape(1,-1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        if self.t_step > self.t_stop_noise:
            self.noise_on = False
        
        self.t_step += 1
        
        if self.t_step % self.update_every == 0 and (len(self.memory) >= self.batch_size):
            #print('LEARNING START!!!')
            experiences = [self.memory.sample() for _ in range(self.n_agents)]
            self.learn(experiences, self.gamma)
                
    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            
            states = states.reshape(-1,self.n_agents,self.state_size).index_select(1, agent_id).squeeze(1)
            next_states = next_states.reshape(-1,self.n_agents,self.state_size).index_select(1, agent_id).squeeze(1)
            
            actions = agent.actor_local(states)
            next_actions = agent.actor_target(next_states)
            
            all_actions.append(actions)
            all_next_actions.append(next_actions)
        
        for i, i_agent in enumerate(self.agents):
            i_agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
            
    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            actions = agent.act(state, noise_weight=self.noise_weight, add_noise=self.noise_on)
            self.noise_weight *= self.noise_decay
            all_actions.append(actions)
            
        return np.array(all_actions).reshape(1,-1) # into (1, 2 x 24)
    
    def save_agents(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_agent_{i}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_agent_{i}.pth')
            
    def reset(self):
        for agent in self.agents:
            agent.reset()

class DDPG():
    def __init__(self, agent_id, model, state_size=24, action_size=2, seed=0, tau=1e-3, lr_actor=1e-4,lr_critic=1e-3,weight_decay=0):
        
        random.seed(seed)
        self.agent_id = agent_id
        self.tau = tau
        
        #CRITIC
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        #ACTOR
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Set weights for local and target actor, respectively, critic the same
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        # Noise
        self.noise = OUNoise(action_size, seed)

        
    def hard_copy_weights(self, target, source):
        for tar, loc in zip(target.parameters(), source.parameters()):
            tar.data.copy_(loc.data)
        
    def act(self, state, noise_weight, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        #print(action[0])
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        
        return np.clip(action,-1,1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences
        '''
        print('states', states.size())
        print('actions', actions.size())
        print('rewards', rewards.size())
        print('next_states', next_states.size())
        print('dones', dones.size())
        '''
        
        # Train critic
        self.critic_optim.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        next_actions = torch.cat(all_next_actions, dim=1).to(device)
        
        with torch.no_grad():
            Q_next = self.critic_target(next_states, next_actions)

        Q_expected = self.critic_local(states, actions)
        
        Q_target = rewards.index_select(1, agent_id) + (gamma * Q_next * (1 - dones.index_select(1, agent_id)))
        #print('KKK',((Y-Q_expected)**2).mean())
        critic_loss = F.mse_loss(Q_expected, Q_target.detach())
        
        critic_loss.backward()
        self.critic_optim.step()
        
        # Train actor
        self.actor_optim.zero_grad()
        actions_pred = [actions if ii == self.agent_id else actions.detach() for ii, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = - self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Update
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
    
    def soft_update(self, local, target, tau):
        for loc, tar in zip(local.parameters(), target.parameters()):
            tar.data.copy_(tau*loc.data + (1.0-tau)*tar.data)
            


# In[ ]:



            

class OUNoise:
    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.2):
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state','action','reward','next_state','done'])
        self.batch_size = batch_size
        random.seed(seed)
        np.random.seed(seed)
        
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