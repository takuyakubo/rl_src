"""
TD(0)学習のデモンストレーション

グリッドワールド環境でTD(0)アルゴリズムの学習過程を可視化する。
TD誤差の推移や価値関数の収束を観察できる。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.gridworld import GridWorldEnvironment
from src.agents import Agent
from src.agents.strategies import TDZeroStrategy
from examples.gridworld_factory import create_gridworld_mdp_core


def run_td_episode(agent: Agent, env: GridWorldEnvironment, max_steps: int = 100) -> List[Dict[str, float]]:
    """
    1エピソードを実行し、TD誤差の情報を収集
    
    Args:
        agent: TD学習エージェント
        env: グリッドワールド環境
        max_steps: 最大ステップ数
        
    Returns:
        各ステップのTD誤差情報のリスト
    """
    observation = env.reset()
    td_errors = []
    
    for _ in range(max_steps):
        # 行動を選択
        available_actions = env.get_available_actions(observation)
        action = agent.get_action(observation, available_actions)
        
        # 環境を1ステップ進める
        next_observation, reward, done, info = env.step(action)
        
        # 経験を作成
        from src.agents.types import Experience
        experience = Experience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done
        )
        
        # TD誤差の情報を取得（更新前）
        if isinstance(agent.strategy, TDZeroStrategy):
            td_error_info = agent.strategy.get_td_error_info(experience)
            td_errors.append(td_error_info)
        
        # エージェントの戦略を更新
        agent.strategy.update(experience)
        
        if done:
            break
            
        observation = next_observation
    
    return td_errors


def visualize_td_learning(
    env: GridWorldEnvironment, 
    strategy: TDZeroStrategy,
    episodes: List[int],
    value_history: Dict[int, Dict[Any, float]]
):
    """TD学習の結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 価値関数の変化をヒートマップで表示
    for idx, episode in enumerate(episodes[:3]):
        ax = axes[idx // 2, idx % 2]
        
        # グリッドの価値を2D配列に変換
        grid_values = np.zeros((env.size, env.size))
        for state, value in value_history[episode].items():
            if isinstance(state, tuple) and len(state) == 2:
                y, x = state
                if 0 <= y < env.size and 0 <= x < env.size:
                    grid_values[y, x] = value
        
        # ヒートマップを描画
        im = ax.imshow(grid_values, cmap='coolwarm', aspect='equal')
        ax.set_title(f'Episode {episode}: Value Function')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 値をセルに表示
        for y in range(env.size):
            for x in range(env.size):
                text = ax.text(x, y, f'{grid_values[y, x]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    # 2. TD誤差の推移
    ax = axes[1, 1]
    ax.set_title('Average TD Error over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average |TD Error|')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """TD(0)学習のデモを実行"""
    print("=== TD(0) Learning Demo ===\n")
    
    # MDPCoreとグリッドワールド環境を作成
    mdp_core = create_gridworld_mdp_core(size=3, goal=(2, 2), stochastic=False)
    env = GridWorldEnvironment(
        mdp_core=mdp_core,
        goal=(2, 2),
        start=(0, 0)
    )
    print(f"Grid World ({env.size}x{env.size})")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print()
    
    # TD(0)戦略を作成
    strategy = TDZeroStrategy(
        epsilon=0.1,
        learning_rate=0.1,
        gamma=0.9,
        initial_value=0.0
    )
    
    # エージェントを作成
    agent = Agent(strategy=strategy)
    
    # 学習を実行
    n_episodes = 100  # デモのため少なめに
    value_history = {}
    td_error_history = []
    episode_rewards = []
    
    print("Training TD(0) agent...")
    start_time = time.time()
    
    for episode in range(n_episodes):
        # エピソードを実行
        td_errors = run_td_episode(agent, env)
        
        # 報酬の合計を計算
        episode_reward = sum(error_info['reward'] for error_info in td_errors)
        episode_rewards.append(episode_reward)
        
        # TD誤差の平均を計算
        if td_errors:
            avg_td_error = np.mean([abs(e['td_error']) for e in td_errors])
            td_error_history.append(avg_td_error)
        
        # 定期的に価値関数を保存
        if episode in [0, 10, 25, 50, 75, 99]:
            value_history[episode] = strategy.get_value_function().copy()
        
        # 進捗を表示
        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_error = np.mean(td_error_history[-100:]) if td_error_history else 0
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Avg Reward = {avg_reward:.2f}, "
                  f"Avg |TD Error| = {avg_error:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # 最終的な価値関数を表示
    print("\nFinal Value Function:")
    final_values = strategy.get_value_function()
    for state, value in sorted(final_values.items()):
        if isinstance(state, tuple) and len(state) == 2:
            print(f"  State {state}: V = {value:.3f}")
    
    # 学習曲線を表示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 報酬の推移
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    # 移動平均を計算
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'{window}-Episode Moving Average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TD誤差の推移
    ax2.plot(td_error_history, alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average |TD Error|')
    ax2.set_title('TD Error Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('td_learning_progress.png')
    print("\nLearning progress saved to 'td_learning_progress.png'")
    
    # 価値関数の変化を可視化
    fig2 = visualize_td_learning(env, strategy, list(value_history.keys()), value_history)
    plt.savefig('td_value_function_evolution.png')
    print("Value function evolution saved to 'td_value_function_evolution.png'")
    
    # 学習後のエージェントの振る舞いを確認
    print("\n=== Learned Policy Demonstration ===")
    print("Executing learned policy (ε=0 for deterministic behavior)...")
    
    # 決定的な方策で実行（探索なし）
    strategy.epsilon = 0.0
    observation = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"\nStarting from: {observation}")
    
    for step in range(50):
        available_actions = env.get_available_actions(observation)
        action = agent.get_action(observation, available_actions)
        next_observation, reward, done, _ = env.step(action)
        
        total_reward += reward
        steps += 1
        
        print(f"Step {step + 1}: State {observation} -> Action {action} -> "
              f"State {next_observation}, Reward: {reward}")
        
        if done:
            print(f"\nGoal reached! Total reward: {total_reward}, Steps: {steps}")
            break
            
        observation = next_observation
    else:
        print(f"\nMax steps reached. Total reward: {total_reward}")


if __name__ == "__main__":
    main()