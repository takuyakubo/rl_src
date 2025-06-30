"""
モンテカルロ法のデモンストレーション

First-visit Monte Carlo、Monte Carlo Control、方策評価の
実装と動作を実証する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from gridworld import GridWorldEnvironment
from gridworld_factory import create_gridworld_mdp_core
from src.agents.strategies.monte_carlo import (
    FirstVisitMonteCarloStrategy,
    MonteCarloControlStrategy,
    MonteCarloEvaluationStrategy
)
from src.agents.strategies.value_iteration import ValueIterationStrategy


def main():
    """モンテカルロ法デモンストレーションのメイン関数"""
    print("=" * 70)
    print("モンテカルロ法のデモンストレーション")
    print("=" * 70)
    
    # 1. First-visit Monte Carlo方法
    demonstrate_first_visit_monte_carlo()
    
    print("\n" + "=" * 70)
    
    # 2. Monte Carlo Control
    demonstrate_monte_carlo_control()
    
    print("\n" + "=" * 70)
    
    # 3. 方策評価
    demonstrate_policy_evaluation()
    
    print("\n" + "=" * 70)
    
    # 4. 価値反復法との比較
    compare_with_value_iteration()


def demonstrate_first_visit_monte_carlo():
    """First-visit Monte Carlo方法のデモ"""
    print("\n=== First-visit Monte Carlo方法 ===")
    
    # 3x3グリッドワールドを作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False, 
        goal=(2, 2),
        move_cost=-0.1,
        goal_reward=1.0,
        random_seed=42
    )
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(2, 2), random_seed=42)
    
    print(f"環境設定:")
    print(f"  グリッドサイズ: {env.size}x{env.size}")
    print(f"  ゴール位置: {env.goal}")
    
    # Monte Carlo戦略を作成
    mc_strategy = FirstVisitMonteCarloStrategy(
        environment=env,
        epsilon=0.3,  # 探索を重視
        gamma=0.9,
        random_seed=42
    )
    
    print(f"\nMonte Carlo設定:")
    print(f"  探索率(ε): {mc_strategy.epsilon}")
    print(f"  割引率(γ): {mc_strategy.gamma}")
    
    # 学習過程を段階的に実行
    learning_stages = [100, 500, 1000, 2000]
    
    print(f"\n=== 学習過程 ===")
    for stage in learning_stages:
        stats = mc_strategy.learn_from_episodes(100)  # 100エピソードずつ学習
        
        print(f"\nエピソード {mc_strategy.episodes_generated}:")
        print(f"  平均エピソード長: {stats['avg_episode_length']:.1f}")
        print(f"  平均報酬: {stats['avg_episode_reward']:.3f}")
        print(f"  学習したQ値の数: {stats['unique_state_actions']}")
        print(f"  総訪問回数: {stats['total_visits']}")
    
    # 学習されたQ値を表示
    print_q_values(mc_strategy, env.size)
    
    # 学習された方策を表示
    print_learned_policy(mc_strategy, env.size)
    
    # 性能をテスト
    test_learned_policy(mc_strategy, env)


def demonstrate_monte_carlo_control():
    """Monte Carlo Controlのデモ"""
    print("\n=== Monte Carlo Control ===")
    
    # 環境を作成
    mdp_core = create_gridworld_mdp_core(
        size=4, 
        stochastic=False, 
        goal=(3, 3),
        random_seed=42
    )
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(3, 3), random_seed=42)
    
    print(f"環境設定:")
    print(f"  グリッドサイズ: {env.size}x{env.size}")
    print(f"  ゴール位置: {env.goal}")
    
    # Monte Carlo Control戦略を作成
    mc_control = MonteCarloControlStrategy(
        environment=env,
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        gamma=0.9,
        random_seed=42
    )
    
    print(f"\nMonte Carlo Control設定:")
    print(f"  初期探索率(ε): {mc_control.initial_epsilon}")
    print(f"  ε減衰率: {mc_control.epsilon_decay}")
    print(f"  最小ε: {mc_control.min_epsilon}")
    
    # 学習過程
    print(f"\n=== 学習過程（探索率減衰あり） ===")
    episode_batch = 200
    num_batches = 10
    
    for batch in range(num_batches):
        stats = mc_control.learn_from_episodes(episode_batch)
        
        print(f"\nバッチ {batch + 1} (エピソード {mc_control.episodes_generated}):")
        print(f"  ε: {stats['epsilon_before']:.3f} → {stats['epsilon_after']:.3f}")
        print(f"  平均報酬: {stats['avg_episode_reward']:.3f}")
        print(f"  平均エピソード長: {stats['avg_episode_length']:.1f}")
        print(f"  Q値数: {stats['unique_state_actions']}")
    
    # 最終方策を表示
    print_learned_policy(mc_control, env.size)
    
    # 性能をテスト
    test_learned_policy(mc_control, env)


def demonstrate_policy_evaluation():
    """Monte Carlo方策評価のデモ"""
    print("\n=== Monte Carlo方策評価 ===")
    
    # 環境を作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False,
        goal=(2, 2),
        random_seed=42
    )
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(2, 2), random_seed=42)
    
    # 評価対象の固定方策を定義（常に右または下に移動）
    def simple_policy(state, available_actions):
        """シンプルな固定方策：右優先、次に下"""
        if 'right' in available_actions:
            return 'right'
        elif 'down' in available_actions:
            return 'down'
        else:
            return np.random.choice(available_actions)
    
    print(f"評価対象方策: 右優先、次に下")
    
    # Monte Carlo方策評価
    mc_eval = MonteCarloEvaluationStrategy(
        environment=env,
        target_policy_func=simple_policy,
        gamma=0.9,
        random_seed=42
    )
    
    # 方策を評価
    print(f"\n=== 方策評価過程 ===")
    evaluation_stages = [500, 1000, 2000, 5000]
    
    for stage in evaluation_stages:
        stats = mc_eval.evaluate_policy(500)  # 500エピソードずつ評価
        
        print(f"\nエピソード {mc_eval.episodes_generated}:")
        print(f"  平均報酬: {stats['avg_episode_reward']:.3f}")
        print(f"  平均エピソード長: {stats['avg_episode_length']:.1f}")
        print(f"  評価したQ値の数: {stats['unique_state_actions']}")
    
    # 価値関数を表示
    value_function = mc_eval.get_value_function()
    print_value_function(value_function, env.size, "評価された価値関数")


def compare_with_value_iteration():
    """価値反復法との比較"""
    print("\n=== 価値反復法との比較 ===")
    
    # 環境を作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False,
        goal=(2, 2),
        random_seed=42
    )
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(2, 2), random_seed=42)
    
    # 価値反復法で最適解を計算
    vi_strategy = ValueIterationStrategy(
        mdp_core=mdp_core,
        gamma=0.9,
        theta=1e-6
    )
    vi_strategy.plan()
    
    print("価値反復法による最適価値関数:")
    vi_value_function = {state: vi_strategy.V.get(state, 0.0) for state in mdp_core.states}
    print_value_function(vi_value_function, env.size, "価値反復法")
    
    # Monte Carlo Controlで学習
    mc_control = MonteCarloControlStrategy(
        environment=env,
        epsilon=0.1,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        gamma=0.9,
        random_seed=42
    )
    
    # 十分に学習
    print(f"\nMonte Carlo Controlで学習中...")
    for _ in range(20):
        mc_control.learn_from_episodes(100)
    
    print(f"総エピソード数: {mc_control.episodes_generated}")
    
    # Monte Carloの価値関数
    mc_value_function = mc_control.get_value_function()
    print_value_function(mc_value_function, env.size, "Monte Carlo Control")
    
    # 価値関数の差を計算
    print(f"\n=== 価値関数の比較 ===")
    print(f"{'状態':<8} {'価値反復法':<12} {'Monte Carlo':<12} {'差分':<8}")
    print("-" * 45)
    
    total_diff = 0.0
    for state in mdp_core.states:
        vi_value = vi_value_function.get(state, 0.0)
        mc_value = mc_value_function.get(state, 0.0)
        diff = abs(vi_value - mc_value)
        total_diff += diff
        
        print(f"{str(state):<8} {vi_value:<12.3f} {mc_value:<12.3f} {diff:<8.3f}")
    
    avg_diff = total_diff / len(mdp_core.states)
    print(f"\n平均絶対誤差: {avg_diff:.4f}")


def print_q_values(strategy, grid_size: int):
    """Q値をテーブル形式で表示"""
    print(f"\n学習されたQ値（抜粋）:")
    
    q_values = strategy.get_all_q_values()
    
    # 各状態の最大Q値を表示
    print(f"{'状態':<8} {'最適行動':<8} {'Q値':<8}")
    print("-" * 30)
    
    for i in range(min(grid_size, 3)):
        for j in range(min(grid_size, 3)):
            state = (i, j)
            state_q_values = {
                action: q_values.get((state, action), 0.0)
                for action in ['up', 'right', 'down', 'left']
            }
            
            if any(q_values.values() for q_values in [state_q_values]):
                best_action = max(state_q_values.keys(), key=lambda a: state_q_values[a])
                best_q = state_q_values[best_action]
                print(f"{str(state):<8} {best_action:<8} {best_q:<8.3f}")


def print_learned_policy(strategy, grid_size: int):
    """学習された方策を表示"""
    print(f"\n学習された方策:")
    
    # 行動の矢印表現
    action_arrows = {
        "up": "↑",
        "right": "→", 
        "down": "↓",
        "left": "←"
    }
    
    # グリッド形式で表示
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            state = (i, j)
            available_actions = ['up', 'right', 'down', 'left']
            
            try:
                action_probs = strategy.get_policy(state, available_actions)
                # 最も確率の高い行動を選択
                best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
                arrow = action_arrows.get(best_action, "?")
                row.append(f"  {arrow}  ")
            except:
                row.append("  ?  ")
        
        print(f"  {''.join(row)}")


def print_value_function(value_function: Dict, grid_size: int, title: str):
    """価値関数をグリッド形式で表示"""
    print(f"\n{title}:")
    
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            value = value_function.get((i, j), 0.0)
            row.append(f"{value:6.3f}")
        print(f"  {' '.join(row)}")


def test_learned_policy(strategy, env: GridWorldEnvironment):
    """学習された方策の性能をテスト"""
    print(f"\n=== 方策の性能テスト ===")
    
    # 複数のエピソードで性能を評価
    num_episodes = 20
    episode_results = []
    
    for episode in range(num_episodes):
        current_state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while not env.is_done() and steps < max_steps:
            available_actions = env.get_available_actions()
            action = strategy.get_action(current_state, available_actions)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            current_state = next_state
        
        episode_results.append({
            "episode": episode + 1,
            "steps": steps,
            "total_reward": total_reward,
            "success": env.is_done()
        })
    
    # 結果を表示
    successful_episodes = [r for r in episode_results if r["success"]]
    success_rate = len(successful_episodes) / num_episodes
    
    print(f"テスト結果 ({num_episodes}エピソード):")
    print(f"  成功率: {success_rate:.1%}")
    
    if successful_episodes:
        avg_steps = np.mean([r["steps"] for r in successful_episodes])
        avg_reward = np.mean([r["total_reward"] for r in successful_episodes])
        print(f"  平均ステップ数: {avg_steps:.1f}")
        print(f"  平均総報酬: {avg_reward:.3f}")
    
    # 詳細結果を表示（最初の3エピソード）
    print(f"\n詳細結果 (最初の3エピソード):")
    for result in episode_results[:3]:
        status = "成功" if result["success"] else "失敗"
        print(f"  エピソード{result['episode']}: {result['steps']}ステップ, "
              f"報酬{result['total_reward']:.3f}, {status}")


if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 70)
    print("モンテカルロ法デモンストレーション完了")
    print("=" * 70)