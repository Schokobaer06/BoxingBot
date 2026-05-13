import random
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import ale_py
# =========================
# SETTINGS
# =========================

EPISODES = 500
MAX_STEPS = 2000

GAMMA = 0.99
LEARNING_RATE = 0.00025

BATCH_SIZE = 32
MEMORY_SIZE = 50000

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

MODEL_NAME = "models/josef-boxing-model.keras"

reward_history = []

# =========================
# ENVIRONMENT
# =========================

env = gym.make("ALE/Boxing-v5")
ACTION_SIZE = int(env.action_space.n)

# =========================
# FRAME STACK
# =========================

frame_stack = deque(maxlen=4)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return (resized / 255.0).astype(np.float32)

def reset_env():
    obs, info = env.reset()
    frame = preprocess(obs)

    frame_stack.clear()
    for _ in range(4):
        frame_stack.append(frame)

    return np.stack(frame_stack, axis=-1)

def step_env(action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    frame = preprocess(obs)
    frame_stack.append(frame)

    next_state = np.stack(frame_stack, axis=-1)

    return next_state, reward, done

# =========================
# REPLAY MEMORY
# =========================

memory = deque(maxlen=MEMORY_SIZE)

# =========================
# MODEL
# =========================

def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(84, 84, 4)),

        layers.Conv2D(32, 8, strides=4, activation='relu'),
        layers.Conv2D(64, 4, strides=2, activation='relu'),
        layers.Conv2D(64, 3, strides=1, activation='relu'),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )

    return model

model = create_model()

# =========================
# TRAINING
# =========================

for episode in range(EPISODES):

    state = reset_env()
    total_reward = 0

    print(f"\n===== Episode {episode+1}/{EPISODES} =====")

    for step in range(MAX_STEPS):

        if step % 100 == 0:
            print(
                f"Episode {episode+1}/{EPISODES} | "
                f"Step {step}/{MAX_STEPS} | "
                f"Reward {total_reward:.1f} | "
                f"Epsilon {EPSILON:.3f}"
            )

        # epsilon-greedy
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, done = step_env(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state

        total_reward += reward

        # TRAINING
        if len(memory) >= 1000 and step % 4 == 0:

            batch = random.sample(memory, BATCH_SIZE)

            states = []
            targets = []

            for s, a, r, ns, d in batch:

                target = model.predict(np.expand_dims(s, axis=0), verbose=0)[0]

                if d:
                    target[a] = r
                else:
                    future_q = np.max(model.predict(np.expand_dims(ns, axis=0), verbose=0)[0])
                    target[a] = r + GAMMA * future_q

                states.append(s)
                targets.append(target)

            model.fit(
                np.array(states),
                np.array(targets),
                epochs=1,
                verbose=0
            )

        if done:
            break

    # epsilon decay
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    reward_history.append(total_reward)

    print(
        f"Episode {episode+1} finished | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {EPSILON:.3f}"
    )

    # save model
    if (episode + 1) % 10 == 0:
        model.save(MODEL_NAME)
        print("Model saved.")

# =========================
# SAVE FINAL MODEL
# =========================

env.close()
model.save(MODEL_NAME)

# =========================
# PLOT TRAINING CURVE
# =========================

plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")

plt.savefig("training_curve.png")
plt.show()

print("Training finished.")