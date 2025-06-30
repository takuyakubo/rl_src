"""
統一的なエージェントアーキテクチャのデモンストレーション

Dependency Injectionによる方策決定戦略の注入を実証し、
Planning, Model-based, Model-freeの3つのアプローチを
同じインターフェースで比較する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Tuple

from gridworld import GridWorldEnvironment
from gridworld_factory import create_gridworld_mdp_core
from src.agents.base import Agent
from src.agents.episode_manager import EpisodeManager
from src.agents.strategies import ValueIterationStrategy, EpsilonGreedyStrategy, RandomStrategy


# この関数は不要になったため削除
# def create_environment_functions(env: GridWorldEnvironment):


def demonstrate_unified_agents():
    """統一的なエージェントアーキテクチャのデモンストレーション"""
    print("=" * 70)
    print("統一的なエージェントアーキテクチャのデモンストレーション")
    print("=" * 70)
    
    # 1. 異なる戦略のエージェントを作成
    agents = create_different_agents()
    
    print("\n" + "=" * 70)
    
    # 2. 各エージェントの性能を比較
    results = compare_agent_performance(agents)
    
    print("\n" + "=" * 70)
    
    # 3. 学習過程の分析
    analyze_learning_process(results)


def create_different_agents():
    """異なる戦略を持つエージェントを作成"""
    print("\n=== 異なる戦略のエージェント作成 ===")
    
    # MDPCoreを作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False, 
        random_seed=42
    )
    
    # 環境設定
    env = GridWorldEnvironment(
        mdp_core=mdp_core,
        goal=(2, 2),
        random_seed=42
    )
    
    print(f"環境設定: {env.size}x{env.size}グリッド, ゴール: {env.goal}")
    
    # 1. プランニングエージェント（価値反復法）
    planning_strategy = ValueIterationStrategy(
        mdp_core=mdp_core,
        gamma=0.9,
        theta=1e-6
    )
    planning_agent = Agent(
        strategy=planning_strategy,
        name="PlanningAgent"
    )
    
    # 2. モデルフリーエージェント（Q学習）
    q_learning_strategy = EpsilonGreedyStrategy(
        epsilon=0.1,
        learning_rate=0.1,
        gamma=0.9
    )
    q_learning_agent = Agent(
        strategy=q_learning_strategy,
        name="QLearningAgent"
    )
    
    # 3. ランダムエージェント（ベースライン）
    random_strategy = RandomStrategy()
    random_agent = Agent(
        strategy=random_strategy,
        name="RandomAgent"
    )
    
    agents = {
        "Planning": planning_agent,
        "Q-Learning": q_learning_agent,
        "Random": random_agent
    }
    
    # エージェント情報を表示
    print(f"\n作成されたエージェント:")
    for name, agent in agents.items():
        print(f"  {name}: {type(agent.strategy).__name__}")
        print(f"    エージェント名: {agent.name}")
        print()
    
    return agents


def compare_agent_performance(agents):
    """エージェントの性能を比較"""
    print("\n=== エージェント性能比較 ===")
    
    # MDPCoreを作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False, 
        random_seed=42
    )
    
    # 環境設定
    env = GridWorldEnvironment(
        mdp_core=mdp_core,
        goal=(2, 2),
        random_seed=42
    )
    
    # 各エージェントでエピソードを実行
    num_episodes = 10
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\n--- {agent_name} のテスト ---")
        
        # エピソード管理を作成
        episode_manager = EpisodeManager()
        episode_results = []
        
        for episode in range(num_episodes):
            # 環境とエージェントをリセット
            observation = env.reset()
            episode_experiences = agent.reset()  # 前回のエピソードの経験を取得
            
            # 前回のエピソードをエピソード管理に追加
            if episode_experiences:
                episode_manager.add_episode_experiences(episode_experiences)
            
            total_reward = 0
            steps = 0
            max_steps = 20
            
            # 最初の行動
            action = agent.step(observation, env.get_available_actions())
            
            while steps < max_steps:
                # 環境で行動を実行
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    # エピソード終了
                    agent.step(next_observation, env.get_available_actions(), reward, done, info)
                    break
                else:
                    # 次の行動を取得
                    action = agent.step(next_observation, env.get_available_actions(), reward, done, info)
                    observation = next_observation
            
            success = env.is_done()
            episode_results.append({
                "episode": episode + 1,
                "steps": steps,
                "total_reward": total_reward,
                "success": success
            })
        
        # 最後のエピソードを追加
        final_experiences = agent.reset()
        if final_experiences:
            episode_manager.add_episode_experiences(final_experiences)
        
        # 結果の集計
        successful_episodes = [r for r in episode_results if r["success"]]
        success_rate = len(successful_episodes) / num_episodes
        
        if successful_episodes:
            avg_steps = np.mean([r["steps"] for r in successful_episodes])
            avg_reward = np.mean([r["total_reward"] for r in successful_episodes])
        else:
            avg_steps = np.mean([r["steps"] for r in episode_results])
            avg_reward = np.mean([r["total_reward"] for r in episode_results])
        
        results[agent_name] = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "episodes": episode_results,
            "episode_manager": episode_manager
        }
        
        print(f"成功率: {success_rate:.1%}")
        print(f"平均ステップ数: {avg_steps:.1f}")
        print(f"平均総報酬: {avg_reward:.3f}")
    
    # 結果の比較表示
    print(f"\n=== 性能比較サマリー ===")
    print(f"{'エージェント':<12} {'成功率':<8} {'平均ステップ':<12} {'平均報酬':<10}")
    print("-" * 50)
    
    for agent_name, result in results.items():
        print(f"{agent_name:<12} {result['success_rate']:<8.1%} "
              f"{result['avg_steps']:<12.1f} {result['avg_reward']:<10.3f}")
    
    return results


def analyze_learning_process(results):
    """学習過程の分析"""
    print("\n=== 学習過程の分析 ===")
    
    # Q学習エージェントの学習過程を詳しく分析
    q_result = results.get("Q-Learning")
    if q_result:
        agent = None
        for agent_name, agent_obj in results.items():
            if agent_name == "Q-Learning":
                # resultsから対応するエージェントを取得
                break
        
        print(f"\n--- Q学習エージェントの学習統計 ---")
        # 戦略情報は別途エージェントから取得
        print(f"学習更新回数: 不明（戦略から直接確認してください）")
        print(f"探索率 (ε): 不明（戦略から直接確認してください）")
    
    # エピソード管理による履歴の比較
    print(f"\n--- エージェント履歴比較（EpisodeManager使用） ---")
    for agent_name, result in results.items():
        episode_manager = result.get("episode_manager")
        if episode_manager:
            print(f"{agent_name}:")
            print(f"  総エピソード数: {episode_manager.get_total_episodes()}")
            if episode_manager.get_total_episodes() > 0:
                print(f"  平均報酬: {episode_manager.get_average_reward():.3f}")
                print(f"  平均エピソード長: {episode_manager.get_average_length():.1f}")
                print(f"  成功率: {episode_manager.get_success_rate():.1%}")
            print()


def demonstrate_strategy_injection():
    """戦略注入の柔軟性を実証"""
    print("\n=== 戦略注入の柔軟性実証 ===")
    
    # MDPCoreを作成
    mdp_core = create_gridworld_mdp_core(
        size=3, 
        stochastic=False, 
        random_seed=42
    )
    
    # 環境設定
    env = GridWorldEnvironment(
        mdp_core=mdp_core,
        goal=(2, 2),
        random_seed=42
    )
    
    # 同じエージェントクラスに異なる戦略を注入
    strategies = {
        "価値反復法": ValueIterationStrategy(
            mdp_core=mdp_core,
            gamma=0.9,
            theta=1e-6
        ),
        "ε-貪欲法": EpsilonGreedyStrategy(
            epsilon=0.2,
            learning_rate=0.1,
            gamma=0.9
        ),
        "ランダム": RandomStrategy()
    }
    
    print(f"同じAgentクラスに異なる戦略を注入:")
    
    for strategy_name, strategy in strategies.items():
        agent = Agent(
            strategy=strategy,
            name=f"Agent({strategy_name})"
        )
        
        print(f"\n{strategy_name}戦略:")
        print(f"  strategy_type: {type(agent.strategy).__name__}")
        print(f"  agent_name: {agent.name}")
        
        # 戦略固有の情報を表示
        strategy_attrs = ['learning_updates', 'epsilon', 'learning_rate', 'theta', 'gamma']
        for attr in strategy_attrs:
            if hasattr(agent.strategy, attr):
                print(f"  {attr}: {getattr(agent.strategy, attr)}")
        
        # 簡単な行動テスト
        test_observation = (0, 0)
        action = agent.step(test_observation, env.get_available_actions(test_observation))
        print(f"  テスト行動: {test_observation} → {action}")


if __name__ == "__main__":
    demonstrate_unified_agents()
    
    print("\n" + "=" * 70)
    
    # 追加デモ：戦略注入の柔軟性
    try:
        demonstrate_strategy_injection()
    except Exception as e:
        print(f"\n戦略注入デモでエラーが発生しました: {e}")
    
    print("\n" + "=" * 70)
    print("統一的なエージェントアーキテクチャデモンストレーション完了")
    print("=" * 70)