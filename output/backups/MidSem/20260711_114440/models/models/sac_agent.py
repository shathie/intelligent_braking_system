import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class GaussianPolicy(nn.Module):
    """Stochastic actor network with Gaussian distribution"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return torch.distributions.Normal(mean, std)
    
    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def to(self, device):
        return super().to(device)


class CriticNetwork(nn.Module):
    """Twin critic networks for Q-value estimation"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class SACAgent:
    """Soft Actor-Critic agent for continuous control"""
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        preferred_device = str(config.get("device", "cuda"))
        if preferred_device.startswith("cuda") and not torch.cuda.is_available():
            preferred_device = "cpu"
        self.device = torch.device(preferred_device)

        def _as_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _as_int(value, default):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)
        
        # Hyperparameters
        self.gamma = _as_float(config.get("gamma", 0.99), 0.99)
        self.tau = _as_float(config.get("tau", 0.005), 0.005)
        self.actor_lr = _as_float(config.get("actor_lr", 3e-4), 3e-4)
        self.critic_lr = _as_float(config.get("critic_lr", 1e-3), 1e-3)
        self.batch_size = _as_int(config.get("batch_size", 256), 256)
        self.buffer_size = _as_int(config.get("buffer_size", 1000000), 1000000)
        
        # Initialize networks
        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        
        # Copy target network weights
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Replay buffer
        self.buffer = deque(maxlen=self.buffer_size)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().detach()
        
        # Training counters
        self.total_steps = 0
        self.learning_steps = 0
    
    def get_action(self, state, deterministic=False):
        """Get action for a given state"""
        if torch.is_tensor(state):
            state_tensor = state.detach().to(self.device).float()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        if deterministic:
            with torch.no_grad():
                dist = self.actor.forward(state_tensor)
                action = dist.mean
        else:
            with torch.no_grad():
                dist = self.actor.forward(state_tensor)
                action = dist.sample()
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor and alpha
        actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.learning_steps += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "learning_steps": self.learning_steps
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        loaded_log_alpha = checkpoint["log_alpha"]
        if torch.is_tensor(loaded_log_alpha):
            loaded_log_alpha = loaded_log_alpha.detach().to(self.device)
        else:
            loaded_log_alpha = torch.tensor(float(loaded_log_alpha), device=self.device)
        self.log_alpha = loaded_log_alpha.requires_grad_(True)
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.alpha = self.log_alpha.exp().detach()
        self.total_steps = checkpoint["total_steps"]
        self.learning_steps = checkpoint["learning_steps"]