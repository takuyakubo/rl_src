"""
価値反復法プランニングのデモンストレーション

MDPCore中心アーキテクチャとValueIterationStrategyを使用して
価値反復法による最適方策の計算過程を実証する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from gridworld import GridWorldEnvironment
from gridworld_factory import create_gridworld_mdp_core
from src.agents.strategies.value_iteration import ValueIterationStrategy


def main():
    """価値反復法デモンストレーションのメイン関数"""
    print("=" * 70)
    print("価値反復法プランニングのデモンストレーション")
    print("=" * 70)
    
    # 1. 基本的な価値反復法の動作確認
    demonstrate_basic_value_iteration()
    
    print("\n" + "=" * 70)
    
    # 2. 異なる設定での比較
    demonstrate_parameter_comparison()
    
    print("\n" + "=" * 70)
    
    # 3. 確率的環境での動作確認
    demonstrate_stochastic_environment()
    
    print("\n" + "=" * 70)
    
    # 4. 方策の実演
    demonstrate_policy_execution()


def demonstrate_basic_value_iteration():
    """基本的な価値反復法の動作確認"""
    print("\n=== 基本的な価値反復法 ===")
    
    # 3x3グリッドワールドのMDPCoreを作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False, 
        goal=(2, 2),
        move_cost=-0.1,
        goal_reward=1.0,
        random_seed=42
    )
    
    # 環境を作成
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(2, 2), random_seed=42)
    
    print(f"環境設定:")
    print(f"  グリッドサイズ: {env.size}x{env.size}")
    print(f"  ゴール位置: {env.goal}")
    print(f"  状態数: {len(mdp_core.states)}")
    print(f"  行動数: {len(mdp_core.actions)}")
    
    # 価値反復法戦略を作成
    strategy = ValueIterationStrategy(
        mdp_core=mdp_core,
        gamma=0.9,
        theta=1e-6,
        max_iterations=1000
    )
    
    print(f"\n価値反復法の設定:")
    print(f"  割引率: {strategy.gamma}")
    print(f"  収束閾値: {strategy.theta}")
    print(f"  最大反復回数: {strategy.max_iterations}")
    
    # 価値反復法を実行（planメソッドで最適方策を計算）
    print(f"\n価値反復法を実行中...")
    optimal_policy = strategy.plan()
    
    # 結果を表示
    print(f"\n学習結果:")
    print(f"  計算された方策のサイズ: {len(optimal_policy)}")
    
    # 価値関数を表示
    print_value_function(strategy, env.size)
    
    # 最適方策を表示
    print_optimal_policy(strategy, env.size)
    
    # 方策の性能をテスト
    test_policy_performance(strategy, env)


def demonstrate_parameter_comparison():
    """異なるパラメータ設定での比較"""
    print("\n=== パラメータ設定の比較 ===")
    
    # 異なる割引率での比較
    gamma_values = [0.5, 0.9, 0.99]
    
    for gamma in gamma_values:
        print(f"\n--- 割引率 γ = {gamma} ---")
        
        mdp_core = create_gridworld_mdp_core(size=3, stochastic=False, random_seed=42)
        
        strategy = ValueIterationStrategy(
            mdp_core=mdp_core,
            gamma=gamma,
            theta=1e-6,
            max_iterations=1000
        )
        
        optimal_policy = strategy.plan()
        
        # ゴール状態の価値を表示
        goal_state = (2, 2)
        goal_value = strategy.V.get(goal_state, 0.0)
        print(f"ゴール状態 {goal_state} の価値: {goal_value:.3f}")
        
        # 開始状態の価値を表示
        start_state = (0, 0)
        start_value = strategy.V.get(start_state, 0.0)
        print(f"開始状態 {start_state} の価値: {start_value:.3f}")


def demonstrate_stochastic_environment():
    """確率的環境での価値反復法"""
    print("\n=== 確率的環境での価値反復法 ===")
    
    # 確率的グリッドワールドのMDPCoreを作成
    mdp_core_stoch = create_gridworld_mdp_core(
        size=3, 
        stochastic=True, 
        random_seed=42
    )
    
    env_stoch = GridWorldEnvironment(mdp_core=mdp_core_stoch, goal=(2, 2), random_seed=42)
    
    print(f"確率的環境設定:")
    print(f"  意図した方向: 80%")
    print(f"  他の方向: 各5%")
    
    strategy_stoch = ValueIterationStrategy(
        mdp_core=mdp_core_stoch,
        gamma=0.9,
        theta=1e-6,
        max_iterations=1000
    )
    
    print(f"\n価値反復法を実行中...")
    optimal_policy_stoch = strategy_stoch.plan()
    
    # 価値関数を表示
    print_value_function(strategy_stoch, env_stoch.size)
    
    # 決定的環境と比較
    print(f"\n--- 決定的環境との比較 ---")
    
    mdp_core_det = create_gridworld_mdp_core(size=3, stochastic=False, random_seed=42)
    strategy_det = ValueIterationStrategy(
        mdp_core=mdp_core_det,
        gamma=0.9,
        theta=1e-6,
        max_iterations=1000
    )
    strategy_det.plan()
    
    # 開始状態の価値を比較
    start_state = (0, 0)
    stochastic_value = strategy_stoch.V.get(start_state, 0.0)
    deterministic_value = strategy_det.V.get(start_state, 0.0)
    
    print(f"開始状態 {start_state} の価値:")
    print(f"  確率的環境: {stochastic_value:.3f}")
    print(f"  決定的環境: {deterministic_value:.3f}")
    print(f"  差分: {abs(stochastic_value - deterministic_value):.3f}")


def demonstrate_policy_execution():
    """学習された方策の実行デモ"""
    print("\n=== 学習された方策の実行デモ ===")
    
    # MDPCoreと環境を作成
    mdp_core = create_gridworld_mdp_core(size=4, goal=(3, 3), stochastic=False, random_seed=42)
    env = GridWorldEnvironment(mdp_core=mdp_core, goal=(3, 3), random_seed=42)
    
    # 価値反復法で最適方策を学習
    strategy = ValueIterationStrategy(
        mdp_core=mdp_core,
        gamma=0.9,
        theta=1e-6
    )
    strategy.plan()
    
    print(f"4x4グリッドでの最適方策実行:")
    print(f"ゴール: {env.goal}")
    
    # エピソードを実行
    current_state = env.reset()
    print(f"\n開始状態: {current_state}")
    
    step = 0
    max_steps = 20
    total_reward = 0
    
    while not env.is_done() and step < max_steps:
        # 現在の状態での行動分布を取得
        available_actions = env.get_available_actions()
        action_probs = strategy.get_policy(current_state, available_actions)
        
        # 最適行動を選択（決定的なので確率1.0の行動）
        action = max(action_probs.keys(), key=lambda a: action_probs[a])
        
        print(f"ステップ{step + 1}: 状態{current_state} → 行動'{action}' (確率: {action_probs[action]:.2f})")
        
        # 行動を実行
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"        → 次状態{next_state}, 報酬{reward:.2f}")
        
        current_state = next_state
        
        if done:
            print(f"🎉 ゴール到達！")
            break
    
    print(f"\nエピソード結果:")
    print(f"  総ステップ数: {step}")
    print(f"  総報酬: {total_reward:.3f}")
    print(f"  成功: {'はい' if env.is_done() else 'いいえ'}")


def print_value_function(strategy: ValueIterationStrategy, grid_size: int):
    """価値関数をグリッド形式で表示"""
    print(f"\n価値関数 (V*):")
    
    # グリッド形式で表示
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            value = strategy.V.get((i, j), 0.0)
            row.append(f"{value:6.3f}")
        print(f"  {' '.join(row)}")


def print_optimal_policy(strategy: ValueIterationStrategy, grid_size: int):
    """最適方策をグリッド形式で表示"""
    print(f"\n最適方策 (π*):")
    
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
            action = strategy.policy.get((i, j), None)
            arrow = action_arrows.get(action, "?")
            row.append(f"  {arrow}  ")
        print(f"  {''.join(row)}")


def test_policy_performance(strategy: ValueIterationStrategy, env: GridWorldEnvironment):
    """学習された方策の性能をテスト"""
    print(f"\n=== 方策の性能テスト ===")
    
    # 複数のエピソードで性能を評価
    num_episodes = 10
    episode_results = []
    
    for episode in range(num_episodes):
        current_state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 20
        
        while not env.is_done() and steps < max_steps:
            # 最適行動を取得
            available_actions = env.get_available_actions()
            action = strategy.get_action(current_state, available_actions)
            
            if action is None:
                print(f"警告: 状態 {current_state} で行動が決定できませんでした")
                break
            
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
    print("価値反復法プランニングデモンストレーション完了")
    print("=" * 70)