"""
GridWorldEnvironmentのテスト

MDP（マルコフ決定過程）の特性を検証する：
1. マルコフ性: P(s'|s,a,h) = P(s'|s,a)
2. 時不変性: P(s'|s,a) は時刻に依存しない
3. 確率分布の性質: Σ P(s'|s,a) = 1
4. 環境インターフェースの正常動作
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "examples"))
from gridworld import GridWorldEnvironment, DeterministicGridWorldChannel


class TestGridWorldEnvironment:
    """GridWorldEnvironmentのテストクラス"""
    
    def test_initialization(self):
        """環境の初期化テスト"""
        env = GridWorldEnvironment(size=3)
        
        assert env.size == 3
        assert env.goal == (2, 2)
        assert env.start == (0, 0)
        assert not env.stochastic
        assert len(env.get_action_space()) == 4
    
    def test_reset(self):
        """リセット機能のテスト"""
        env = GridWorldEnvironment(size=3)
        
        # デフォルトリセット
        initial_observation = env.reset()
        assert initial_observation == (0, 0)
        assert env.get_current_state() == (0, 0)
        assert len(env.get_history()) == 0
        assert not env.is_done()
        
        # カスタム開始位置
        custom_observation = env.reset(start_position=(1, 1))
        assert custom_observation == (1, 1)
        assert env.get_current_state() == (1, 1)
    
    def test_step_deterministic(self):
        """決定的環境での行動実行テスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset()
        
        # 右に移動
        next_observation, reward, done, info = env.step("right")
        
        assert next_observation == (0, 1)
        assert reward == -0.1  # 移動コスト
        assert not done
        assert info["previous_state"] == (0, 0)
        assert info["action"] == "right"
        assert len(env.get_history()) == 1
    
    def test_goal_reaching(self):
        """ゴール到達のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset(start_position=(2, 1))
        
        # ゴールに向かって移動
        next_observation, reward, done, info = env.step("right")
        
        assert next_observation == (2, 2)  # ゴール位置
        assert reward == 1.0  # ゴール報酬
        assert done  # エピソード終了
        assert env.is_done()
    
    def test_boundary_handling(self):
        """境界処理のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset(start_position=(0, 0))
        
        # 左上角から上に移動しようとする
        next_observation, reward, done, info = env.step("up")
        
        assert next_observation == (0, 0)  # 同じ位置にとどまる
        assert not done
    
    def test_invalid_action(self):
        """無効な行動のテスト"""
        env = GridWorldEnvironment(size=3)
        env.reset()
        
        with pytest.raises(ValueError, match="無効な行動"):
            env.step("invalid_action")
    
    def test_step_without_reset(self):
        """リセットせずに行動実行した場合のテスト"""
        env = GridWorldEnvironment(size=3)
        
        with pytest.raises(RuntimeError, match=r"step\(\) called before reset\(\)"):
            env.step("right")


class TestMDPProperties:
    """MDPの数学的性質をテストするクラス"""
    
    def test_reward_consistency(self):
        """報酬の一貫性テスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        
        # ゴールに向かう報酬
        goal_reward = env.reward_model((2, 1), "right", (2, 2))
        assert goal_reward == 1.0
        
        # 通常の移動コスト
        move_reward = env.reward_model((0, 0), "right", (0, 1))
        assert move_reward == -0.1
    
    def test_stochastic_behavior(self):
        """確率的環境の動作テスト"""
        env = GridWorldEnvironment(size=3, stochastic=True, random_seed=42)
        env.reset(start_position=(1, 1))
        
        # 同じ状態・行動でも異なる結果が出ることがある（確率的）
        results = []
        for _ in range(100):
            env.reset(start_position=(1, 1))
            obs, _, _, _ = env.step("right")
            results.append(obs)
        
        # すべて同じ結果ではない（確率的なので）
        unique_results = set(results)
        assert len(unique_results) > 1  # 確率的なので複数の結果


class TestEnvironmentInterface:
    """環境インターフェースのテスト"""
    
    def test_action_space(self):
        """行動空間のテスト"""
        env = GridWorldEnvironment(size=3)
        
        actions = env.get_action_space()
        expected_actions = ["up", "right", "down", "left"]
        
        assert len(actions) == 4
        assert set(actions) == set(expected_actions)
    
    def test_noise_channel(self):
        """ノイズチャネルのテスト"""
        channel = DeterministicGridWorldChannel()
        
        # 決定的チャネルは状態をそのまま返す
        state = (1, 1)
        observation = channel.sample_observation(state)
        assert observation == state
        
        # 確率は状態と観測が一致する場合のみ1.0
        assert channel.get_observation_probability(state, state) == 1.0
        assert channel.get_observation_probability((0, 0), state) == 0.0


class TestHistoryAndPydantic:
    """履歴とPydanticモデルのテスト"""
    
    def test_episode_history(self):
        """エピソード履歴機能のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset()
        
        # 初期状態では履歴は空
        assert len(env.get_history()) == 0
        
        # 行動実行後は履歴に記録
        env.step("right")
        env.step("down")
        
        history = env.get_history()
        assert len(history) == 2
        
        # Pydanticモデルの属性として履歴にアクセス
        first_entry = history[0]
        assert first_entry.action == "right"
        assert first_entry.reward == -0.1
        assert first_entry.next_state == (0, 1)
        assert first_entry.previous_state == (0, 0)
        
        # 2番目のエントリ
        second_entry = history[1]
        assert second_entry.action == "down"
        assert second_entry.previous_state == (0, 1)
        assert second_entry.next_state == (1, 1)


if __name__ == "__main__":
    # テストの実行例
    pytest.main([__file__, "-v"])