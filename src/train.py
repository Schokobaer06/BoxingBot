import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import DQN
from replay_memory import ReplayMemory
from utils import reset_stack, stack_frames
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Einstellungen

EPISODES = 10

MAX_STEPS = 5

BATCH_SIZE = 64

GAMMA = 0.99

LEARNING_RATE = 0.00025

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

MEMORY_SIZE = 100000

TARGET_UPDATE = 10

# Envirionment

env = gym.make("ALE/Boxing-v5")

action_size = env.action_space.n # type: ignore

# Model

policy_net = DQN(action_size).to(device)

target_net = DQN(action_size).to(device)

target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()

optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

memory = ReplayMemory(MEMORY_SIZE)

# Training

reward_history = []

for episode in range(EPISODES):

    obs, info = env.reset()

    state = reset_stack(obs)

    total_reward = 0

    for step in range(MAX_STEPS):

        # epsilon-greedy
        if random.random() < EPSILON:
            action = env.action_space.sample()

        else:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)

            action = torch.argmax(q_values).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = stack_frames(next_obs)

        memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward # type: ignore

        # trainieren, wenn genug Daten da sind
        if len(memory) >= BATCH_SIZE:

            batch = memory.sample(BATCH_SIZE)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(
                np.array(states),
                dtype=torch.float32
            ).to(device)

            actions = torch.tensor(actions).to(device)

            rewards = torch.tensor(
                rewards,
                dtype=torch.float32
            ).to(device)

            next_states = torch.tensor(
                np.array(next_states),
                dtype=torch.float32
            ).to(device)

            dones = torch.tensor(
                dones,
                dtype=torch.float32
            ).to(device)

            current_q = policy_net(states).gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()

            with torch.no_grad():

                next_actions = policy_net(
                    next_states
                ).argmax(1)

                next_q_values = target_net(
                    next_states
                ).gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze()

                target_q = rewards + (
                    GAMMA * next_q_values * (1 - dones)
                )
            loss = criterion(current_q, target_q)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        if done:
            break

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    reward_history.append(total_reward)
    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(
            policy_net.state_dict()
        )
        print("Target network updated.")

    print(
        f"Episode {episode+1} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {EPSILON:.3f}"
    )
    

# Model speichern

torch.save(
    policy_net.state_dict(),
    "models/boxing_model.pth"
)

plt.plot(reward_history)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")

plt.savefig("plots/training_curve.png")

plt.show()

print("Training finished.")

env.close()