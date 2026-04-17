# Snake AI ML Agent: Deep Q-Learning & Double DQN
A high-performance Snake AI agent trained using Reinforcement Learning (PyTorch). This project implements and compares standard DQN against Double DQN in dynamic environments with randomized layouts and static obstacles.

<img width="1000" height="800" alt="graph_0" src="https://github.com/user-attachments/assets/3ee7340f-8440-4454-b12b-56e5c5829356" />
---
<img width="1000" height="800" alt="graph_1" src="https://github.com/user-attachments/assets/cb21c684-6839-4ba7-a058-29029aede8d0" />
---
<img width="1000" height="800" alt="graph_2" src="https://github.com/user-attachments/assets/a3fe52fb-c1fb-4449-b466-f78e90cfe03c" />
---
<img width="1000" height="800" alt="graph_3" src="https://github.com/user-attachments/assets/7172c282-1608-45df-aea6-63e20ef6c5ae" />

## Highlights
- **Architecture**: Linear Q-Network (11 features input).
- **Techniques**: Experience Replay, Target Networks (DDQN), Huber Loss, Epsilon-Greedy.
- **Environments**: Randomized map layouts (Octagon, Cross, Maze) and incremental obstacles.
- **Analysis**: Thread-safe centralized logging and automated evaluation metrics.

## Experimental Results (Final Benchmarks)
Performance measured over 100 training episodes across 4 parallel agents:

| Metric | Basic DQN | Double DQN | Improve % |
| :--- | :--- | :--- | :--- |
| **All-Games Mean Score** | 8.39 | 10.88 | +29.7% |
| **Stable Mean (Ep > 50)** | 15.58 | 19.61 | **+25.8%** |
| **Max Score Achieved** | 53 | 70 | +32.0% |

> **Conclusion**: The Double DQN architecture significantly reduced Q-value overestimation, leading to more stable policies and a 25.8% increase in score during the stabilized training phase.

## Neural Network & State Space
- **Inputs (11)**: Directional danger (straight, L, R), current direction (one-hot), relative food position.
- **Hidden layer**: 256 neurons (ReLU).
- **Optimization**: Adam, Huber Loss, Gamma 0.99, Batch 128.

## Project Structure
- `agent.py`: Orchestrator for parallel training and evaluation.
- `game.py`: Custom Pygame environment with randomized layouts.
- `model.py`: DQN/DDQN implementation and trainer logic.
- `model/`: Directory containing trained weights:
  - `explorer_0.pth` & `explorer_1.pth`: Basic DQN agents.
  - `trained_2.pth` & `trained_3.pth`: Double DQN agents (high performance).
- `training_stats.csv`: Raw data from the finalized experimental run.
- `architecture.txt`: Detailed technical report and ML analysis.

## How to Run
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train Agents**: `python agent.py` (Starts 4 parallel instances).
3. **Evaluate Model**: `python agent.py --eval 2 true` (Validates model 2 in greedy mode).

---
*Developed for research and evaluation of model-free reinforcement learning in constrained spatial environments.*




