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
from src.rl_src.environments import GridWorldEnvironment


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
        assert len(env.get_state_space()) == 9
    
    def test_reset(self):
        """リセット機能のテスト"""
        env = GridWorldEnvironment(size=3)
        
        # デフォルトリセット
        initial_state = env.reset()
        assert initial_state == (0, 0)
        assert env.get_current_state() == (0, 0)
        assert len(env.get_history()) == 0
        assert not env.is_done()
        
        # カスタム開始位置
        custom_start = env.reset(start_position=(1, 1))
        assert custom_start == (1, 1)
        assert env.get_current_state() == (1, 1)
    
    def test_step_deterministic(self):
        """決定的環境での行動実行テスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset()
        
        # 右に移動
        next_state, reward, done, info = env.step("right")
        
        assert next_state == (0, 1)
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
        next_state, reward, done, info = env.step("right")
        
        assert next_state == (2, 2)  # ゴール位置
        assert reward == 1.0  # ゴール報酬
        assert done  # エピソード終了
        assert env.is_done()
    
    def test_boundary_handling(self):
        """境界処理のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset(start_position=(0, 0))
        
        # 左上角から上に移動しようとする
        next_state, reward, done, info = env.step("up")
        
        assert next_state == (0, 0)  # 同じ位置にとどまる
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
        
        with pytest.raises(ValueError, match="環境がリセットされていません"):
            env.step("right")


class TestMDPProperties:
    """MDPの数学的性質をテストするクラス"""
    
    def test_markov_property_deterministic(self):
        """決定的環境でのマルコフ性テスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        
        # 同じ状態・行動の組み合わせは常に同じ遷移
        test_state = (1, 1)
        test_action = "right"
        
        transition_model = env.get_transition_model()
        prob_dist = transition_model[test_state][test_action]
        
        # 決定的環境では確率1で単一の次状態
        assert len(prob_dist) == 1
        assert sum(prob_dist.values()) == 1.0
        
        expected_next = (1, 2)
        assert prob_dist[expected_next] == 1.0
    
    def test_markov_property_stochastic(self):
        """確率的環境でのマルコフ性テスト"""
        env = GridWorldEnvironment(size=3, stochastic=True, random_seed=42)
        
        test_state = (1, 1)
        test_action = "right"
        
        transition_model = env.get_transition_model()
        prob_dist = transition_model[test_state][test_action]
        
        # 確率の和が1
        assert abs(sum(prob_dist.values()) - 1.0) < 1e-10
        
        # 意図した方向の確率が最大
        intended_next = (1, 2)
        assert prob_dist[intended_next] >= 0.8
    
    def test_time_invariance(self):
        """時不変性のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=True, random_seed=42)
        
        test_state = (1, 1)
        test_action = "right"
        
        # 初期の遷移確率を取得
        initial_prob = env.get_transition_model()[test_state][test_action].copy()
        
        # 複数ステップ実行後も同じ確率分布
        env.reset()
        for _ in range(10):
            if not env.is_done():
                env.step("right")
        
        # 遷移確率は変化しない
        current_prob = env.get_transition_model()[test_state][test_action]
        assert initial_prob == current_prob
    
    def test_reward_consistency(self):
        """報酬の一貫性テスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        
        reward_model = env.get_reward_model()
        
        # ゴールに向かう行動は正の報酬
        goal_state = (2, 1)
        goal_action = "right"
        assert reward_model[goal_state][goal_action] == 1.0
        
        # 通常の移動は負のコスト
        normal_state = (0, 0)
        normal_action = "right"
        assert reward_model[normal_state][normal_action] == -0.1
    
    def test_state_space_completeness(self):
        """状態空間の完全性テスト"""
        env = GridWorldEnvironment(size=3)
        
        state_space = env.get_state_space()
        transition_model = env.get_transition_model()
        
        # 全ての状態が遷移モデルに含まれている
        for state in state_space:
            assert state in transition_model
            
            for action in env.get_action_space():
                assert action in transition_model[state]
                
                # 各行動に対して確率分布が定義されている
                prob_dist = transition_model[state][action]
                assert abs(sum(prob_dist.values()) - 1.0) < 1e-10
    
    def test_stochastic_vs_deterministic(self):
        """確率的vs決定的環境の比較テスト"""
        det_env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
        stoch_env = GridWorldEnvironment(size=3, stochastic=True, random_seed=42)
        
        test_state = (1, 1)
        test_action = "right"
        
        det_prob = det_env.get_transition_model()[test_state][test_action]
        stoch_prob = stoch_env.get_transition_model()[test_state][test_action]
        
        # 決定的環境は単一の次状態
        assert len(det_prob) == 1
        
        # 確率的環境は複数の次状態
        assert len(stoch_prob) > 1


class TestEnvironmentInterface:
    """環境インターフェースのテスト"""
    
    def test_action_space(self):
        """行動空間のテスト"""
        env = GridWorldEnvironment(size=3)
        
        actions = env.get_action_space()
        expected_actions = ["up", "right", "down", "left"]
        
        assert len(actions) == 4
        assert set(actions) == set(expected_actions)
    
    def test_state_space(self):
        """状態空間のテスト"""
        env = GridWorldEnvironment(size=3)
        
        states = env.get_state_space()
        
        assert len(states) == 9  # 3x3 grid
        assert (0, 0) in states
        assert (2, 2) in states
        assert (3, 3) not in states  # 範囲外
    
    def test_render_modes(self):
        """描画モードのテスト"""
        env = GridWorldEnvironment(size=3)
        env.reset()
        
        # humanモード（出力のみ、例外なし）
        env.render(mode="human")
        
        # rgb_arrayモード
        rgb_array = env.render(mode="rgb_array")
        assert rgb_array is not None
        assert isinstance(rgb_array, np.ndarray)
        assert len(rgb_array.shape) == 3  # height, width, channels
        
        # 無効なモード
        with pytest.raises(ValueError, match="サポートされていない描画モード"):
            env.render(mode="invalid")
    
    def test_episode_analysis(self):
        """エピソード分析機能のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        env.reset(start_position=(2, 1))
        
        # ゴールに到達
        env.step("right")
        
        analysis = env.analyze_episode()
        
        assert analysis["total_steps"] == 1
        assert analysis["total_reward"] == 1.0
        assert analysis["reached_goal"] is True
        assert analysis["final_position"] == (2, 2)
        assert len(analysis["trajectory"]) == 1


class TestMDPDemonstrations:
    """MDP概念の実証機能テスト"""
    
    def test_markov_demonstration(self):
        """マルコフ性実証機能のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=True)
        
        # 例外なく実行できることを確認
        env.demonstrate_markov_property()
    
    def test_time_invariance_demonstration(self):
        """時不変性実証機能のテスト"""
        env = GridWorldEnvironment(size=3, stochastic=False)
        
        # 例外なく実行できることを確認
        env.demonstrate_time_invariance()
    
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
        
        # 履歴の形式確認
        first_entry = history[0]
        assert first_entry["action"] == "right"
        assert first_entry["reward"] == -0.1
        assert first_entry["observation"] == (0, 1)


if __name__ == "__main__":
    # テストの実行例
    pytest.main([__file__, "-v"])