import gymnasium as gym
import ale_py
import torch
import numpy as np
import cv2
import imageio

from model import DQN
from utils import reset_stack, stack_frames

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================

env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")

action_size = env.action_space.n # type: ignore

model = DQN(action_size).to(device)
model.load_state_dict(torch.load("boxing_model.pth", map_location=device))
model.eval()

# =========================
# EVAL SETTINGS
# =========================

EPISODES = 5
all_rewards = []

frames = []

# =========================
# EVALUATION LOOP
# =========================

for episode in range(EPISODES):

    obs, info = env.reset()
    state = reset_stack(obs)

    total_reward = 0
    done = False

    episode_frames = []

    while not done:

        frame = env.render()
        episode_frames.append(frame)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(state_tensor)

        action = torch.argmax(q_values).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        state = stack_frames(next_obs)

        total_reward += reward # type: ignore

    all_rewards.append(total_reward)

    frames.extend(episode_frames)

    print(f"Episode {episode+1} reward: {total_reward}")

# =========================
# RESULTS
# =========================

avg_reward = sum(all_rewards) / len(all_rewards)

print("\n========================")
print("Evaluation finished")
print(f"Average reward (5 games): {avg_reward:.2f}")
print("========================")

# =========================
# SAVE VIDEO
# =========================

video_path = "evaluation_video.mp4"

imageio.mimsave(video_path, frames, fps=30)

print(f"Video saved to {video_path}")

env.close()