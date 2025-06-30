"""
価値反復法のテストコード

ValueIterationAgentの機能をテストし、正しく動作することを確認する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from typing import Tuple

from src.agents import ValueIterationAgent
from examples.gridworld import GridWorldEnvironment


class TestValueIterationAgent:
    """価値反復法エージェントのテストクラス"""
    
    def test_simple_grid_world(self):
        """2x2の簡単なグリッドワールドでの基本動作テスト"""
        # 2x2グリッドワールドを設定
        states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        actions = ["up", "right", "down", "left"]
        goal_state = (1, 1)
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            """決定的な遷移確率"""
            expected_next = self._get_next_state_2x2(state, action)
            return 1.0 if next_state == expected_next else 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            """報酬関数：ゴール到達で1、移動で-0.1"""
            if next_state == goal_state:
                return 1.0
            else:
                return -0.1
        
        # 価値反復法エージェントを作成
        agent = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            theta=1e-6,
            random_seed=42
        )
        
        # 学習前の状態をテスト
        assert not agent.is_trained()
        
        # 価値反復法を実行
        optimal_values, optimal_policy = agent.solve()
        
        # 学習後の状態をテスト
        assert agent.is_trained()
        assert len(optimal_values) == len(states)
        assert len(optimal_policy) == len(states)
        
        # ゴール状態の価値が正の値であることを確認
        goal_value = agent.get_value(goal_state)
        assert goal_value > 5.0  # ゴール状態は高い価値を持つはず
        
        # 開始状態からの最適経路をテスト
        start_state = (0, 0)
        action = agent.get_action(start_state)
        assert action in actions
        
        # 学習統計を確認
        stats = agent.get_learning_stats()
        assert stats["converged"] is True
        assert stats["iterations"] > 0
        assert stats["final_delta"] < agent.theta
    
    def test_stochastic_environment(self):
        """確率的環境での動作テスト"""
        states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        actions = ["right", "down"]
        goal_state = (1, 1)
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            """確率的な遷移確率"""
            if state == goal_state:  # ゴール状態では動かない
                return 1.0 if next_state == goal_state else 0.0
            
            # 意図した方向80%、他の方向10%ずつ
            intended_next = self._get_next_state_2x2(state, action)
            if next_state == intended_next:
                return 0.8
            elif next_state == state:  # 同じ場所にとどまる
                return 0.2
            else:
                return 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            return 1.0 if next_state == goal_state else -0.1
        
        agent = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            theta=1e-6,
            random_seed=42
        )
        
        optimal_values, optimal_policy = agent.solve()
        
        # 確率的環境でも収束することを確認
        assert agent.is_trained()
        stats = agent.get_learning_stats()
        assert stats["converged"] is True
    
    def test_discount_factor_effect(self):
        """割引率の効果をテスト"""
        states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        actions = ["right", "down"]
        goal_state = (1, 1)
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            expected_next = self._get_next_state_2x2(state, action)
            return 1.0 if next_state == expected_next else 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            return 1.0 if next_state == goal_state else -0.1
        
        # 異なる割引率で比較
        gamma_values = [0.5, 0.9, 0.99]
        start_state = (0, 0)
        
        values_by_gamma = {}
        
        for gamma in gamma_values:
            agent = ValueIterationAgent(
                states=states,
                actions=actions,
                transition_prob=transition_prob,
                reward_func=reward_func,
                gamma=gamma,
                theta=1e-6,
                random_seed=42
            )
            
            agent.solve()
            values_by_gamma[gamma] = agent.get_value(start_state)
        
        # 割引率が高いほど開始状態の価値が高くなることを確認
        assert values_by_gamma[0.99] > values_by_gamma[0.9] > values_by_gamma[0.5]
    
    def test_convergence_criteria(self):
        """収束基準のテスト"""
        states = [(0, 0), (0, 1)]
        actions = ["right"]
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            if state == (0, 0) and action == "right":
                return 1.0 if next_state == (0, 1) else 0.0
            else:
                return 1.0 if next_state == state else 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            return 1.0 if next_state == (0, 1) else 0.0
        
        # 厳しい収束基準
        agent_strict = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            theta=1e-8,  # 厳しい
            random_seed=42
        )
        
        # 緩い収束基準
        agent_loose = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            theta=1e-3,  # 緩い
            random_seed=42
        )
        
        agent_strict.solve()
        agent_loose.solve()
        
        stats_strict = agent_strict.get_learning_stats()
        stats_loose = agent_loose.get_learning_stats()
        
        # 厳しい基準の方が多くの反復が必要
        assert stats_strict["iterations"] >= stats_loose["iterations"]
    
    def test_online_update_raises_error(self):
        """オンライン更新メソッドがエラーを発生させることをテスト"""
        states = [(0, 0), (0, 1)]
        actions = ["right"]
        
        def transition_prob(state, action, next_state):
            return 1.0 if next_state == (0, 1) else 0.0
        
        def reward_func(state, action, next_state):
            return 1.0
        
        agent = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9
        )
        
        # updateメソッドはNotImplementedErrorを発生させるべき
        with pytest.raises(NotImplementedError):
            agent.update((0, 0), "right", 1.0, (0, 1), False)
    
    def test_get_action_before_training_raises_error(self):
        """学習前にget_actionを呼ぶとエラーになることをテスト"""
        states = [(0, 0), (0, 1)]
        actions = ["right"]
        
        def transition_prob(state, action, next_state):
            return 1.0
        
        def reward_func(state, action, next_state):
            return 1.0
        
        agent = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9
        )
        
        # 学習前にget_actionを呼ぶとRuntimeErrorが発生するべき
        with pytest.raises(RuntimeError):
            agent.get_action((0, 0))
    
    def test_q_value_computation(self):
        """Q値の計算が正しいことをテスト"""
        states = [(0, 0), (0, 1)]
        actions = ["right", "stay"]
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            if action == "right":
                return 1.0 if next_state == (0, 1) else 0.0
            else:  # stay
                return 1.0 if next_state == state else 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            return 1.0 if next_state == (0, 1) else 0.0
        
        agent = ValueIterationAgent(
            states=states,
            actions=actions,
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            random_seed=42
        )
        
        agent.solve()
        
        # (0,0)からrightのQ値は正の値になるはず
        q_right = agent.get_q_value((0, 0), "right")
        q_stay = agent.get_q_value((0, 0), "stay")
        
        assert q_right > q_stay  # rightの方が良い行動
        assert q_right > 0  # 正の報酬が得られる
    
    def test_with_gridworld_environment(self):
        """GridWorldEnvironmentとの統合テスト"""
        # 実際のGridWorldEnvironmentを使用
        env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
        
        def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            expected_next = env._get_next_state(state, action)
            return 1.0 if next_state == expected_next else 0.0
        
        def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
            return env.reward_model(state, action, next_state)
        
        agent = ValueIterationAgent(
            states=env._state_space,
            actions=env.get_action_space(),
            transition_prob=transition_prob,
            reward_func=reward_func,
            gamma=0.9,
            theta=1e-6,
            random_seed=42
        )
        
        optimal_values, optimal_policy = agent.solve()
        
        # 環境との統合が正しく動作することを確認
        assert agent.is_trained()
        assert len(optimal_values) == len(env._state_space)
        
        # 実際にエピソードを実行してテスト
        env.reset()
        total_reward = 0
        steps = 0
        max_steps = 20
        
        while not env.is_done() and steps < max_steps:
            current_state = env.get_current_state()
            action = agent.get_action(current_state)
            
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # ゴールに到達できることを確認
        assert env.is_done()
        assert total_reward > 0  # 正の総報酬
    
    def _get_next_state_2x2(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """2x2グリッド用の状態遷移関数"""
        i, j = state
        
        if action == "up":
            next_i = max(0, i - 1)
            next_j = j
        elif action == "right":
            next_i = i
            next_j = min(1, j + 1)
        elif action == "down":
            next_i = min(1, i + 1)
            next_j = j
        elif action == "left":
            next_i = i
            next_j = max(0, j - 1)
        else:
            next_i, next_j = i, j
        
        return (next_i, next_j)


def test_value_iteration_properties():
    """価値反復法の数学的性質をテスト"""
    # 最適性の必要条件：ベルマン最適方程式を満たすか
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    actions = ["right", "down"]
    goal_state = (1, 1)
    
    def transition_prob(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        if state == goal_state:
            return 1.0 if next_state == goal_state else 0.0
        
        if action == "right":
            next_state_expected = (state[0], min(1, state[1] + 1))
        else:  # down
            next_state_expected = (min(1, state[0] + 1), state[1])
        
        return 1.0 if next_state == next_state_expected else 0.0
    
    def reward_func(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        return 1.0 if next_state == goal_state else -0.1
    
    agent = ValueIterationAgent(
        states=states,
        actions=actions,
        transition_prob=transition_prob,
        reward_func=reward_func,
        gamma=0.9,
        theta=1e-8,
        random_seed=42
    )
    
    agent.solve()
    
    # ベルマン最適方程式の確認：V*(s) = max_a Q*(s,a)
    for state in states:
        if state == goal_state:
            continue
        
        state_value = agent.get_value(state)
        q_values = [agent.get_q_value(state, action) for action in actions]
        max_q = max(q_values)
        
        # 許容誤差内でベルマン最適方程式を満たすか
        assert abs(state_value - max_q) < 1e-6


if __name__ == "__main__":
    # 個別にテストを実行
    test_instance = TestValueIterationAgent()
    
    print("価値反復法のテストを実行中...")
    
    try:
        test_instance.test_simple_grid_world()
        print("✓ 基本動作テスト")
        
        test_instance.test_stochastic_environment()
        print("✓ 確率的環境テスト")
        
        test_instance.test_discount_factor_effect()
        print("✓ 割引率効果テスト")
        
        test_instance.test_convergence_criteria()
        print("✓ 収束基準テスト")
        
        test_instance.test_online_update_raises_error()
        print("✓ オンライン更新エラーテスト")
        
        test_instance.test_get_action_before_training_raises_error()
        print("✓ 学習前エラーテスト")
        
        test_instance.test_q_value_computation()
        print("✓ Q値計算テスト")
        
        test_instance.test_with_gridworld_environment()
        print("✓ GridWorld統合テスト")
        
        test_value_iteration_properties()
        print("✓ 数学的性質テスト")
        
        print("\n全てのテストが成功しました！")
        
    except Exception as e:
        print(f"\nテストが失敗しました: {e}")
        raise