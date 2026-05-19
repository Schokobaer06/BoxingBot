import random
from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# =========================
# SETTINGS
# =========================

EPISODES = 100
MAX_STEPS = 500

GAMMA = 0.99
LEARNING_RATE = 0.00025

BATCH_SIZE = 32
MEMORY_SIZE = 50000

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

MODEL_NAME = "boxing-model.keras"

# =========================
# ENVIRONMENT
# =========================

env = gym.make("ALE/Boxing-v5")

ACTION_SIZE = int(env.action_space.n)

# =========================
# PREPROCESSING
# =========================

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized.astype(np.float32)

# =========================
# REPLAY MEMORY
# =========================

memory = deque(maxlen=MEMORY_SIZE)

# =========================
# MODEL
# =========================

def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(84, 84, 1)),

        layers.Conv2D(32, 8, strides=4, activation='relu'),
        layers.Conv2D(64, 4, strides=2, activation='relu'),
        layers.Conv2D(64, 3, strides=1, activation='relu'),

        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss='mse'
    )

    return model

model = create_model()

# =========================
# TRAINING
# =========================

for episode in range(EPISODES):
    print(f"\n===== Episode {episode + 1}/{EPISODES} =====")
    obs, info = env.reset()

    state = preprocess(obs)
    state = np.expand_dims(state, axis=-1)

    total_reward = 0

    for step in range(MAX_STEPS):
        print(f"Step: {step}")
        # epsilon-greedy
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            q_values = model.predict(
                np.expand_dims(state, axis=0),
                verbose=0
            )

            action = np.argmax(q_values[0])

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = preprocess(next_obs)
        next_state = np.expand_dims(next_state, axis=-1)

        memory.append(
            (state, action, reward, next_state, done)
        )

        state = next_state

        total_reward += reward

        # training
        if len(memory) >= BATCH_SIZE:

            batch = random.sample(memory, BATCH_SIZE)

            states = []
            targets = []

            for s, a, r, ns, d in batch:

                target = model.predict(
                    np.expand_dims(s, axis=0),
                    verbose=0
                )[0]

                if d:
                    target[a] = r
                else:
                    future_q = np.max(
                        model.predict(
                            np.expand_dims(ns, axis=0),
                            verbose=0
                        )[0]
                    )

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

    print(
        f"Episode {episode + 1} | "
        f"Reward: {total_reward} | "
        f"Epsilon: {EPSILON:.3f}"
    )

    # autosave
    if (episode + 1) % 10 == 0:
        model.save(MODEL_NAME)
        print("Model saved.")

env.close()

model.save(MODEL_NAME)

print("Training finished.")