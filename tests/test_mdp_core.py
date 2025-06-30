"""
MDPCoreアーキテクチャのテスト

環境から独立したMDPCoreを使用したValueIterationStrategyのテスト。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gridworld import GridWorldEnvironment
from src.agents.strategies import ValueIterationStrategy
from src.agents.base import Agent
from src.environments.mdp_core import MDPCore


def test_mdp_core_extraction():
    """環境からMDPCoreを抽出するテスト"""
    print("=== MDPCore抽出テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    mdp_core = env.get_mdp_core()
    
    print(f"MDPCore状態数: {len(mdp_core.states)}")
    print(f"MDPCore行動数: {len(mdp_core.actions)}")
    print(f"MDPCore状態空間: {mdp_core.states}")
    print(f"MDPCore行動空間: {mdp_core.actions}")
    
    # MDPCoreの内容が環境と一致することを確認
    assert mdp_core.states == env.states, "状態空間が一致しません"
    assert mdp_core.actions == env.actions, "行動空間が一致しません"
    
    # 遷移モデルと報酬モデルが同じ結果を返すことを確認
    state = (0, 0)
    action = "right"
    next_state = (0, 1)
    
    env_transition_prob = env.transition_model(state, action, next_state)
    core_transition_prob = mdp_core.transition_model(state, action, next_state)
    print(f"環境遷移確率: {env_transition_prob}")
    print(f"Core遷移確率: {core_transition_prob}")
    assert env_transition_prob == core_transition_prob, "遷移確率が一致しません"
    
    env_reward = env.reward_model(state, action, next_state)
    core_reward = mdp_core.reward_model(state, action, next_state)
    print(f"環境報酬: {env_reward}")
    print(f"Core報酬: {core_reward}")
    assert env_reward == core_reward, "報酬が一致しません"
    
    print("MDPCore抽出テスト: 成功")


def test_mdp_core_validation():
    """MDPCoreの整合性検証テスト"""
    print("\n=== MDPCore整合性検証テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    mdp_core = env.get_mdp_core()
    
    try:
        is_valid = mdp_core.validate()
        print(f"MDPCore整合性: {is_valid}")
        assert is_valid, "MDPCore validation failed"
        print("MDPCore整合性検証テスト: 成功")
    except Exception as e:
        print(f"MDPCore整合性検証でエラー: {e}")
        raise


def test_value_iteration_with_mdp_core():
    """ValueIterationStrategyのMDPCore使用テスト"""
    print("\n=== ValueIterationStrategy + MDPCore テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    mdp_core = env.get_mdp_core()
    
    # ValueIterationStrategyの作成（環境を直接参照しない）
    strategy = ValueIterationStrategy(
        mdp_core=mdp_core,
        gamma=0.9,
        theta=1e-6
    )
    
    # エージェントの作成
    agent = Agent(strategy=strategy, name="TestPlanningAgent")
    
    # 環境とは独立してテスト
    observation = (0, 0)
    available_actions = ["up", "right", "down", "left"]
    
    # 行動選択テスト
    action = agent.step(observation, available_actions)
    print(f"観測 {observation} での選択行動: {action}")
    
    assert action in available_actions, f"Invalid action: {action}"
    
    # 戦略が計画済みであることを確認
    assert strategy._is_planned, "戦略が計画されていません"
    print("ValueIterationStrategy + MDPCore テスト: 成功")


def test_mdp_core_independence():
    """MDPCoreの環境からの独立性テスト"""
    print("\n=== MDPCore独立性テスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    mdp_core = env.get_mdp_core()
    
    # 元の環境を削除しても MDPCore は動作する
    del env
    
    # MDPCoreが単独で機能することを確認
    state = (0, 0)
    action = "right"
    next_state = (0, 1)
    
    prob = mdp_core.transition_model(state, action, next_state)
    reward = mdp_core.reward_model(state, action, next_state)
    converted_state = mdp_core.observation_to_state((0, 0))
    
    print(f"独立したMDPCore - 遷移確率: {prob}")
    print(f"独立したMDPCore - 報酬: {reward}")
    print(f"独立したMDPCore - 状態変換: {converted_state}")
    
    assert prob == 1.0, f"Expected 1.0, got {prob}"
    assert reward == -0.1, f"Expected -0.1, got {reward}"
    assert converted_state == (0, 0), f"Expected (0,0), got {converted_state}"
    
    print("MDPCore独立性テスト: 成功")


if __name__ == "__main__":
    test_mdp_core_extraction()
    test_mdp_core_validation()
    test_value_iteration_with_mdp_core()
    test_mdp_core_independence()
    
    print("\n" + "=" * 50)
    print("全てのMDPCoreテストが成功しました！")
    print("環境と戦略の疎結合が実現されています。")
    print("=" * 50)