import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.001
LR_CRITIC = 0.002
BUFFER_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODES = 200
MAX_STEPS = 200

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return torch.tanh(self.output(x)) * self.max_action

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += noise * np.random.normal(0, 1, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Critic loss
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Training
env = gym.make("Pendulum-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action)

# Set up video recording
from gym.wrappers import RecordVideo
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

episode_rewards = []
for episode in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0
    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        episode_reward += reward

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{MAX_EPISODES}, Reward: {episode_reward:.2f}")

    if np.mean(episode_rewards[-10:]) > -200:
        print("Solved!")
        break

env.close()
