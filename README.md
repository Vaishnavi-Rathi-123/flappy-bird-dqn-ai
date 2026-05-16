# 🐦 Flappy Bird AI using Deep Learning & Reinforcement Learning

A Deep Q-Network (DQN) based AI agent that learns to play Flappy Bird using **Deep Learning (DL)** and **Reinforcement Learning (RL)** techniques.

The agent interacts with the environment, learns from rewards and penalties, and gradually improves its gameplay performance over thousands of episodes.

---

# 📌 Project Overview

This project combines:

- **Deep Learning**
- **Reinforcement Learning**
- **Neural Networks**
- **Q-Learning**
- **Experience Replay**
- **Target Networks**

The AI agent was trained for approximately **25,000 episodes** to learn stable gameplay behavior.

---

# 🧠 Concepts Used

## 🔹 Deep Learning (DL)

Used Neural Networks to approximate Q-values:

- Input Layer → Environment States
- Hidden Layers → Learning patterns
- Output Layer → Q-values for actions

Framework Used:
- PyTorch

---

## 🔹 Reinforcement Learning (RL)

The agent learns by:
1. Taking actions
2. Receiving rewards/penalties
3. Updating policy based on experience

RL Techniques Used:
- Deep Q-Network (DQN)
- Epsilon-Greedy Exploration
- Experience Replay
- Target Network Synchronization

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- YAML

---

The model was trained for approximately **25,000 episodes**.

During training:
- Experience replay memory stores gameplay experiences
- Neural network weights are updated
- Best model gets saved automatically
- Epsilon decays gradually for better exploitation

---

# 🧩 Features

- Deep Q-Network (DQN)
- Deep Learning based Q-value prediction
- Reinforcement Learning environment interaction
- Experience Replay Memory
- Target Network
- GPU / CUDA / MPS Support
- Model Saving & Loading
- Reward Logging

---

# 📈 Training Output Example

```text
Episode: 24531 | Reward: 18.0 | Epsilon: 0.05
Episode: 24532 | Reward: 21.0 | Epsilon: 0.05
```

---

# 📚 Reinforcement Learning Workflow

```text
Environment → State → Agent → Action → Reward → Learning Update
```

---

# 🔮 Future Improvements

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- TensorBoard Visualization
- Reward Shaping
- Better Neural Architecture

---

# 👩‍💻 Author

**Vaishnavi Rathi**

AI/ML Diploma Student  
Passionate about Deep Learning, Reinforcement Learning, and AI Systems
