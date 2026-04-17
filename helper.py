import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

def plot(scores, rewards, filename='training_graph.png'):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # score lpot
    ax1.set_title('Score per Episode')
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Raw Score', alpha=0.3, color='blue')
    
    if len(scores) > 0:
        mean_scores = [np.mean(scores[max(0, i-50):i+1]) for i in range(len(scores))]
        ax1.plot(mean_scores, label='Moving Avg (50)', color='red', linewidth=2)
        ax1.text(len(scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}", color='red')

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Cumulative Reward & Stability')
    ax2.set_xlabel('Games')
    ax2.set_ylabel('Reward')
    ax2.plot(rewards, label='Raw Reward', alpha=0.3, color='green')
    
    if len(rewards) > 0:
        mean_rewards = [np.mean(rewards[max(0, i-50):i+1]) for i in range(len(rewards))]
        if len(rewards) > 10:
            stds = [np.std(rewards[max(0, i-20):i+1]) for i in range(len(rewards))]
            ax2.fill_between(range(len(rewards)), 
                            np.array(mean_rewards) - np.array(stds), 
                            np.array(mean_rewards) + np.array(stds), 
                            color='green', alpha=0.1, label='Volatility (Std)')
        
        ax2.plot(mean_rewards, label='Moving Avg (50)', color='darkgreen', linewidth=2)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
