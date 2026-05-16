import random
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:

    def __init__(self, param_set):

        self.param_set = param_set   

        with open(param_set, "r") as f:
            all_param_set = yaml.safe_load(f) 
            params = all_param_set["param_set"]
                 
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"] 
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")



    def run(self, is_training=True, render=False):
    
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] # Input dimensions
        num_actions = env.action_space.n # Output dimensions
        
        policy_dqn = DQN(num_states, num_actions).to(device)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)

            # Copy the weights and biases from policy_dqn to target_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha) 
            best_reward = float('-inf') 

        else:
            # Best Model or Policy Load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval() # Set the model to evaluation mode

        for episode in itertools.count():

            if episode >= 100000:
                break

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # Explore: random action
                    action = torch.tensor(action, dtype=torch.long).to(device)

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                # Create Tensors
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

                
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state

            if is_training:
                print(f"Episode: {episode+1} | Reward: {episode_reward} | Epsilon: {epsilon}")

            if not is_training:
                print(f"Episode: {episode+1} | Reward: {episode_reward}")

            if is_training:
                # Epsilon Decay
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

                if episode_reward > best_reward:
                    log_msg = f"Best Reward : {episode_reward} for Episode: {episode+1}\n"

                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_msg)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            if is_training and len(memory) > self.mini_batch_size:
                # Get Sample
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(policy_dqn, target_dqn, mini_batch)

                # Sync the Networks
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
            

            # env.close() - Manually Stop
            # env.close()

    def optimize(self, policy_dqn, target_dqn, mini_batch):
        # Get Experiences => Batch Train
        # Unpack mini-batch
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        # Convert to tensors
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).long().to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.stack(rewards).float().to(device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(device)

        # Calculate target Q-values
        with torch.no_grad():   
            target_q = rewards + self.gamma * (1 - terminations) * target_dqn(next_states).max(dim=1)[0]

        # Calculate y_pred i.e. Q-value for current policy
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    # Parse Command line Inputs
    parser = argparse.ArgumentParser(description="Train or Test the DQN Agent for Flappy Bird")
    parser.add_argument('Hyperparameters', help='')
    parser.add_argument('--train', action='store_true', help='Training Mode')
    parser.add_argument('--test', action='store_true', help='Testing Mode')

    args = parser.parse_args()

    dql = Agent(param_set=args.Hyperparameters)

    if args.train:
        dql.run(is_training=True)

    elif args.test:
        dql.run(is_training=False, render=True)

    else:
        print("Use --train or --test")