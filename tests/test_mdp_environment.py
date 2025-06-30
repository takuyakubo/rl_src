"""
MDPEnvironmentの実装テスト

GridWorldEnvironmentがMDPEnvironmentを正しく実装していることを確認する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gridworld import GridWorldEnvironment
from src.agents.strategies import ValueIterationStrategy
from src.agents.base import Agent


def test_mdp_environment():
    """MDPEnvironmentインターフェースのテスト"""
    print("=== MDPEnvironment インターフェーステスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    
    # MDPプロパティのテスト
    print(f"状態数: {len(env.states)}")
    print(f"行動数: {len(env.actions)}")
    print(f"状態空間: {env.states}")
    print(f"行動空間: {env.actions}")
    
    assert len(env.states) == 9, f"Expected 9 states, got {len(env.states)}"
    assert len(env.actions) == 4, f"Expected 4 actions, got {len(env.actions)}"
    
    # transition_modelのテスト
    prob = env.transition_model((0, 0), "right", (0, 1))
    print(f"P((0,1) | (0,0), right) = {prob}")
    assert prob == 1.0, f"Expected 1.0, got {prob}"
    
    # reward_modelのテスト
    reward = env.reward_model((0, 0), "right", (0, 1))
    print(f"R((0,0), right, (0,1)) = {reward}")
    assert reward == -0.1, f"Expected -0.1, got {reward}"
    
    print("MDPEnvironment インターフェーステスト: 成功")


def test_value_iteration_with_mdp():
    """ValueIterationStrategyのMDPEnvironment統合テスト"""
    print("\n=== ValueIterationStrategy + MDPEnvironment 統合テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    
    # ValueIterationStrategyの作成
    strategy = ValueIterationStrategy(
        mdp_env=env,
        gamma=0.9,
        theta=1e-6
    )
    
    # エージェントの作成
    agent = Agent(strategy=strategy, name="TestPlanningAgent")
    
    # 環境をリセット
    observation = env.reset()
    print(f"初期観測: {observation}")
    
    # 行動選択テスト
    available_actions = env.get_available_actions()
    action = agent.step(observation, available_actions)
    print(f"選択された行動: {action}")
    
    # 環境でステップ実行
    next_observation, reward, done, info = env.step(action)
    print(f"次の観測: {next_observation}")
    print(f"報酬: {reward}")
    print(f"終了: {done}")
    
    assert action in available_actions, f"Invalid action: {action}"
    print("ValueIterationStrategy + MDPEnvironment 統合テスト: 成功")


def test_mdp_validation():
    """MDP整合性検証テスト"""
    print("\n=== MDP整合性検証テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    
    try:
        is_valid = env.validate_mdp()
        print(f"MDP整合性: {is_valid}")
        assert is_valid, "MDP validation failed"
        print("MDP整合性検証テスト: 成功")
    except Exception as e:
        print(f"MDP整合性検証でエラー: {e}")
        raise


if __name__ == "__main__":
    test_mdp_environment()
    test_value_iteration_with_mdp()
    test_mdp_validation()
    
    print("\n" + "=" * 50)
    print("全てのMDPEnvironmentテストが成功しました！")
    print("=" * 50)