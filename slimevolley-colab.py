"""
Temporal Difference Learning for Slime Volleyball

This notebook implements and trains a TD learning agent for the Slime Volleyball game.
"""

import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
import pickle



import slimevolleygym

class TDAgent:
    """
    Temporal Difference Learning Agent for Slime Volleyball
    
    Uses a neural network to approximate the state-value function.
    Implements TD(λ) learning with eligibility traces.
    """
    def __init__(self, input_dim=12, hidden_dim=64, lr=0.001, gamma=0.99, lambda_=0.9, epsilon=0.1):
        """
        Initialize the TD agent.
        
        Args:
            input_dim: Dimension of the state input (12 for SlimeVolley)
            hidden_dim: Number of neurons in the hidden layer
            lr: Learning rate
            gamma: Discount factor
            lambda_: TD(λ) parameter for eligibility traces
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.action_space = 3  # forward, backward, jump
        
        # Initialize weights
        # Input -> Hidden
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        
        # Hidden -> Output (value function)
        self.w2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # Policy network (Hidden -> Actions)
        self.w_policy = np.random.randn(hidden_dim, self.action_space) * 0.1
        self.b_policy = np.zeros(self.action_space)
        
        # Eligibility traces
        self.reset_eligibility_traces()
        
        # For improved exploration
        self.action_probs = np.ones(self.action_space) / self.action_space
        
    def reset_eligibility_traces(self):
        """Reset eligibility traces to zero."""
        self.e_w1 = np.zeros_like(self.w1)
        self.e_b1 = np.zeros_like(self.b1)
        self.e_w2 = np.zeros_like(self.w2)
        self.e_b2 = np.zeros_like(self.b2)
        self.e_w_policy = np.zeros_like(self.w_policy)
        self.e_b_policy = np.zeros_like(self.b_policy)
    
    def forward(self, state):
        """
        Forward pass through the neural network.
        
        Args:
            state: The current state observation
            
        Returns:
            hidden: Hidden layer activations
            value: Estimated state value
            action_probs: Action probabilities
        """
        # Scale inputs to be in a reasonable range
        state = state / 10.0
        
        # Hidden layer with tanh activation
        hidden = np.tanh(np.dot(state, self.w1) + self.b1)
        
        # Value output
        value = np.dot(hidden, self.w2) + self.b2
        
        # Action probabilities with softmax
        logits = np.dot(hidden, self.w_policy) + self.b_policy
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)
        
        return hidden, value, action_probs
    
    def predict(self, state):
        """
        Predict action based on current policy.
        
        Args:
            state: Current state observation
            
        Returns:
            action: [forward, backward, jump] binary action array
        """
        _, _, action_probs = self.forward(state)
        self.action_probs = action_probs  # Save for learning
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action
            action = np.zeros(3)
            idx = np.random.randint(0, 3)
            action[idx] = 1
        else:
            # Greedy action with respect to policy
            action = np.zeros(3)
            max_idx = np.argmax(action_probs)
            action[max_idx] = 1
            
            # Sometimes choose random other actions to enforce exploration
            if np.random.random() < 0.05:  # 5% chance to explore other actions
                other_idx = np.random.choice([i for i in range(3) if i != max_idx])
                action_prime = np.zeros(3)
                action_prime[other_idx] = 1
                return action_prime
                
        return action
    
    def update(self, state, next_state, reward, done):
        """
        Update the network weights using TD(λ) learning.
        
        Args:
            state: Current state
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        # Forward pass for current state
        state = state / 10.0  # Scale state
        hidden, value, _ = self.forward(state)
        
        # Target value (bootstrapped if not done)
        if done:
            target = reward
        else:
            next_state = next_state / 10.0  # Scale next state
            _, next_value, _ = self.forward(next_state)
            target = reward + self.gamma * next_value
        
        # TD error
        td_error = target - value
        
        # Backpropagation for value network
        # Output layer gradients
        d_value = -td_error  # MSE derivative
        d_w2 = np.outer(hidden, d_value)
        d_b2 = d_value
        
        # Hidden layer gradients
        d_hidden = np.dot(d_value, self.w2.T)
        d_hidden = d_hidden * (1 - hidden**2)  # tanh derivative
        d_w1 = np.outer(state, d_hidden)
        d_b1 = d_hidden
        
        # Update eligibility traces
        self.e_w2 = self.gamma * self.lambda_ * self.e_w2 + d_w2
        self.e_b2 = self.gamma * self.lambda_ * self.e_b2 + d_b2
        self.e_w1 = self.gamma * self.lambda_ * self.e_w1 + d_w1
        self.e_b1 = self.gamma * self.lambda_ * self.e_b1 + d_b1
        
        # Update weights using eligibility traces
        self.w2 -= self.lr * self.e_w2
        self.b2 -= self.lr * self.e_b2
        self.w1 -= self.lr * self.e_w1
        self.b1 -= self.lr * self.e_b1
        
        # Policy update using advantage
        advantage = td_error  # Use TD error as advantage estimate
        
        # Update policy based on action probabilities and advantage
        action_idx = np.argmax(self.action_probs)
        
        # Create a target distribution: increase probability of better actions
        target_probs = self.action_probs.copy()
        if advantage > 0:
            # Increase probability of taken action if advantage is positive
            target_probs[action_idx] += 0.1
            target_probs /= target_probs.sum()  # Normalize
        else:
            # Decrease probability of taken action if advantage is negative
            target_probs[action_idx] = max(0.01, target_probs[action_idx] - 0.1)
            target_probs /= target_probs.sum()  # Normalize
            
        # Update policy weights
        d_policy = -(target_probs - self.action_probs)
        d_w_policy = np.outer(hidden, d_policy)
        d_b_policy = d_policy
        
        # Update policy eligibility traces
        self.e_w_policy = self.gamma * self.lambda_ * self.e_w_policy + d_w_policy
        self.e_b_policy = self.gamma * self.lambda_ * self.e_b_policy + d_b_policy
        
        # Update policy weights
        self.w_policy -= self.lr * 0.01 * self.e_w_policy  # Lower learning rate for policy
        self.b_policy -= self.lr * 0.01 * self.e_b_policy
        
        return td_error
    
    def decay_epsilon(self, factor=0.999):
        """Decay exploration rate."""
        self.epsilon *= factor
        self.epsilon = max(0.01, self.epsilon)  # Don't go below 1%
        
    def save(self, filename):
        """Save the agent's parameters."""
        params = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
            'w_policy': self.w_policy,
            'b_policy': self.b_policy,
            'hyperparams': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'lr': self.lr,
                'gamma': self.gamma,
                'lambda_': self.lambda_,
                'epsilon': self.epsilon
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
            
    def load(self, filename):
        """Load the agent's parameters."""
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        
        self.w1 = params['w1']
        self.b1 = params['b1']
        self.w2 = params['w2']
        self.b2 = params['b2']
        self.w_policy = params['w_policy']
        self.b_policy = params['b_policy']
        
        # Optionally update hyperparameters
        hyperparams = params.get('hyperparams', {})
        self.input_dim = hyperparams.get('input_dim', self.input_dim)
        self.hidden_dim = hyperparams.get('hidden_dim', self.hidden_dim)
        self.lr = hyperparams.get('lr', self.lr)
        self.gamma = hyperparams.get('gamma', self.gamma)
        self.lambda_ = hyperparams.get('lambda_', self.lambda_)
        self.epsilon = hyperparams.get('epsilon', self.epsilon)

def train_td_agent(env, agent, episodes=1000, max_steps=3000, eval_freq=50):
    """
    Train a TD agent in the SlimeVolley environment.
    
    Args:
        env: The SlimeVolley gym environment
        agent: The TD agent to train
        episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        eval_freq: Frequency of evaluation episodes
        
    Returns:
        dict: Training metrics
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'td_errors': [],
        'eval_scores': []
    }
    
    # Training loop
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        td_errors = []
        
        # Reset eligibility traces at the start of each episode
        agent.reset_eligibility_traces()
        
        for step in range(max_steps):
            # Select action according to policy
            action = agent.predict(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update agent (learn)
            td_error = agent.update(state, next_state, reward, done)
            td_errors.append(td_error)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_epsilon(0.995)
        
        # Record metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['episode_lengths'].append(step + 1)
        metrics['td_errors'].append(np.mean(np.abs(td_errors)))
        
        # Evaluate periodically
        if episode % eval_freq == 0:
            eval_score = evaluate_agent(env, agent, n_episodes=10, render=False)
            metrics['eval_scores'].append(eval_score)
            print(f"Episode {episode}, Eval score: {eval_score:.3f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint periodically
        if episode % 100 == 0 and episode > 0:
            agent.save(f"td_agent_checkpoint_{episode}.pkl")
    
    # Final evaluation
    eval_score = evaluate_agent(env, agent, n_episodes=50, render=False)
    print(f"Final evaluation: {eval_score:.3f}")
    
    # Save final model
    agent.save("td_agent_final.pkl")
    
    return metrics

def evaluate_agent(env, agent, n_episodes=10, render=False):
    """
    Evaluate a trained agent.
    
    Args:
        env: The SlimeVolley gym environment
        agent: The agent to evaluate
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        float: Average score
    """
    scores = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # During evaluation, use the greedy policy
            prev_epsilon = agent.epsilon
            agent.epsilon = 0.0
            
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if render:
                env.render()
                time.sleep(0.02)
            
            agent.epsilon = prev_epsilon
            
        scores.append(total_reward)
    
    return np.mean(scores)

def plot_metrics(metrics):
    """Plot training metrics."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    axs[0, 0].plot(metrics['episode_rewards'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    
    # Plot episode lengths
    axs[0, 1].plot(metrics['episode_lengths'])
    axs[0, 1].set_title('Episode Lengths')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    
    # Plot TD errors
    axs[1, 0].plot(metrics['td_errors'])
    axs[1, 0].set_title('Average TD Errors')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('TD Error')
    
    # Plot evaluation scores
    eval_episodes = np.arange(0, len(metrics['episode_rewards']), 50)[:len(metrics['eval_scores'])]
    axs[1, 1].plot(eval_episodes, metrics['eval_scores'])
    axs[1, 1].set_title('Evaluation Scores')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def test_against_baseline(env, agent, n_episodes=100, render=False):
    """Test the agent against the baseline policy."""
    baseline = slimevolleygym.BaselinePolicy()
    
    wins = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # Agent action (right side)
            agent_action = agent.predict(obs)
            
            # Baseline action (left side)
            baseline_obs = env.unwrapped.agent_left.getObservation()
            baseline_action = baseline.predict(baseline_obs)
            
            # Step with both actions
            obs, reward, done, info = env.step(agent_action, baseline_action)
            
            ep_reward += reward
            
            if render:
                env.render()
                time.sleep(0.02)
        
        total_reward += ep_reward
        if ep_reward > 0:
            wins += 1
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {ep_reward}")
    
    win_rate = wins / n_episodes
    avg_reward = total_reward / n_episodes
    
    print(f"Results against baseline:")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average reward: {avg_reward:.4f}")
    
    return win_rate, avg_reward

# Main training script
if __name__ == "__main__":
    # Create environment
    env = gym.make("SlimeVolley-v0")
    
    # Create and train the TD agent
    agent = TDAgent(
        input_dim=12,  # State dimension for SlimeVolley
        hidden_dim=64,  # Size of hidden layer
        lr=0.001,       # Learning rate
        gamma=0.99,     # Discount factor
        lambda_=0.9,    # TD(λ) parameter
        epsilon=0.2     # Initial exploration rate
    )
    
    # Train the agent
    metrics = train_td_agent(
        env=env,
        agent=agent,
        episodes=5000,   # Total training episodes
        eval_freq=50     # Evaluate every 50 episodes
    )
    
    # Plot training metrics
    plot_metrics(metrics)
    
    # Test against baseline policy
    test_against_baseline(env, agent, n_episodes=100, render=True)