import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt
import csv
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001

class Agent:

    def __init__(self, phase=2, use_double_dqn=False):
        self.n_games = 0
        self.epsilon = 0 # Aleatoriedade
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.target_model = None
        self.total_steps = 0
        if use_double_dqn:
            self.target_model = Linear_QNet(11, 256, 3)
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, target_model=self.target_model)
        self.phase = phase
        self.use_double_dqn = use_double_dqn

    def sync_target_model(self):
        if self.target_model:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.food.x < game.head.x,  # fl
            game.food.x > game.head.x,  # fr
            game.food.y < game.head.y,  # fu
            game.food.y > game.head.y   # fd
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        
        is_random = False
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            is_random = True
        else:
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move, is_random, prediction.detach().numpy(), torch.max(prediction).item()


def train(agent_id=0, use_double_dqn=False, lock=None):
    plot_scores = []
    plot_rewards = []
    total_score = 0
    record = 0
    agent = Agent(phase=2, use_double_dqn=use_double_dqn)
    title = f"Agent {agent_id} - {'Double DQN' if use_double_dqn else 'Basic DQN'}"
    game = SnakeGameAI(phase=2, title=title)
    
    current_ep_reward = 0
    
    print(f"Training started for {title}...")

    total_frames = 0
    avg_frames = 0
    
    csv_filename = 'training_stats.csv'
    
    if lock:
        with lock:
            if not os.path.exists(csv_filename):
                with open(csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Episode', 'AgentID', 'Type', 'Score', 'Reward', 'Steps'])
    else:
        if not os.path.exists(csv_filename):
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'AgentID', 'Type', 'Score', 'Reward', 'Steps'])

    while True:
        state_old = agent.get_state(game)

        final_move, is_random, q_vals, max_q = agent.get_action(state_old)

        debug_info = {
            "Agent": f"{agent_id}",
            "Type": "Trained (DDQN)" if use_double_dqn else "Explorer (DQN)",
            "Game": f"{agent.n_games}",
            "Max Q": f"{max_q:.2f}",
            "Seed": f"{game.seed_val}",
            "Mode": "Scanning (Random)" if is_random else "Optimized (Model)",
            "Prediction": f"Str:{q_vals[0]:.2f} R:{q_vals[1]:.2f} L:{q_vals[2]:.2f}",
            "Action": ["STRAIGHT", "RIGHT", "LEFT"][np.argmax(final_move)],
            "Frames": f"{game.frame_iteration}",
            "Avg Survival": f"{avg_frames:.1f} frames"
        }

        reward, done, score = game.play_step(final_move, debug_info=debug_info)
        current_ep_reward += reward
        state_new = agent.get_state(game)
        agent.total_steps += 1

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # sincronizar apos 1000 passos
        if use_double_dqn and agent.total_steps % 1000 == 0:
            agent.sync_target_model()

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            total_frames += game.frame_iteration
            plot_rewards.append(current_ep_reward)
            current_ep_reward = 0
            game.reset()
            agent.n_games += 1
            avg_frames = total_frames / agent.n_games
            agent.train_long_memory()

            if score > record:
                record = score
                prefix = "trained" if use_double_dqn else "explorer"
                agent.model.save(file_name=f'{prefix}_{agent_id}.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            line = [agent.n_games, agent_id, ("DDQN" if use_double_dqn else "DQN"), score, current_ep_reward, game.frame_iteration]
            if lock:
                with lock:
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(line)
            else:
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)

            plot_scores.append(score)
            total_score += score
            plot(plot_scores, plot_rewards, filename=f'graph_{agent_id}.png')
            
            if agent.phase == 1 and record >= 20:
                agent.phase = 2
                game.phase = 2
                print("Transitioning to Phase 2: Variable Seeds.")
            elif agent.phase == 2 and record >= 50:
                agent.phase = 3
                game.phase = 3
                game.reset()
                print("Transitioning to Phase 3: Incremental Obstacles.")
            
    print(f"Agent {agent_id} finished training.")

def evaluate(agent_id, use_double_dqn, num_episodes=100):
    print(f"Starting Evaluation for Agent {agent_id} (Greedy Policy)...")
    agent = Agent(phase=2, use_double_dqn=use_double_dqn)
    model_path = f'model/{"trained" if use_double_dqn else "explorer"}_{agent_id}.pth'
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    game = SnakeGameAI(phase=2, title=f"EVALUATION - Agent {agent_id}")
    
    scores = []
    rewards = []
    
    for i in range(num_episodes):
        game.reset()
        done = False
        ep_reward = 0
        while not done:
            state = agent.get_state(game)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state0)
            move = [0,0,0]
            move[torch.argmax(prediction).item()] = 1
            
            reward, done, score = game.play_step(move)
            ep_reward += reward
            
        scores.append(score)
        rewards.append(ep_reward)
        print(f"Eval Episode {i+1}: Score {score}, Reward {ep_reward:.2f}")

    print("\n--- EVALUATION RESULTS ---")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print("--------------------------")

if __name__ == '__main__':
    from multiprocessing import Process, Lock
    import sys
    
    if '--eval' in sys.argv:
        aid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        ddqn = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
        evaluate(aid, ddqn)
    else:
        agents_to_run = [
            (0, False),
            (1, False),
            (2, True),
            (3, True)
        ]
        
        lock = Lock()
        processes = []
        print("Launching AI Command Center...")
        
        for aid, ddqn in agents_to_run:
            p = Process(target=train, args=(aid, ddqn, lock))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
