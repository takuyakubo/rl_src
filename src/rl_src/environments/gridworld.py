"""
グリッドワールド環境の実装

マルコフ決定過程の具体例として、シンプルなグリッドワールドを実装。
Zenn記事での説明と同じ仕様で、実際に動作するコードを提供する。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any
from .base import MDPEnvironment


class GridWorldEnvironment(MDPEnvironment):
    """
    グリッドワールド環境
    
    N×Nのグリッド上でエージェントが移動し、ゴールを目指す環境。
    マルコフ決定過程として完全に定式化されている。
    
    MDP構成要素:
    - S: グリッド上の座標 (i, j) の集合
    - A: {"up", "right", "down", "left"} の行動集合
    - P: 状態遷移確率（決定的または確率的）
    - R: 報酬関数（ゴール到達で正の報酬、移動で負のコスト）
    """
    
    def __init__(
        self,
        size: int = 3,
        goal: Tuple[int, int] = (2, 2),
        start: Optional[Tuple[int, int]] = None,
        stochastic: bool = False,
        move_cost: float = -0.1,
        goal_reward: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        グリッドワールドを初期化
        
        Args:
            size: グリッドのサイズ (size × size)
            goal: ゴール位置の座標
            start: 開始位置（Noneの場合は(0,0)）
            stochastic: 確率的遷移を使用するか
            move_cost: 移動時のコスト（負の値）
            goal_reward: ゴール到達時の報酬
            random_seed: 乱数シード
        """
        super().__init__()
        
        self.size = size
        self.goal = goal
        self.start = start if start is not None else (0, 0)
        self.stochastic = stochastic
        self.move_cost = move_cost
        self.goal_reward = goal_reward
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # === MDP構成要素の構築 ===
        
        # 状態集合 S
        self._state_space = [(i, j) for i in range(size) for j in range(size)]
        
        # 行動集合 A
        self._action_space = ["up", "right", "down", "left"]
        self._action_effects = {
            "up": (-1, 0),
            "right": (0, 1),
            "down": (1, 0),
            "left": (0, -1)
        }
        
        # 状態遷移モデル P(s'|s,a) を構築
        self._transition_model = self._build_transition_model()
        
        # 報酬モデル R(r|s,a) を構築
        self._reward_model = self._build_reward_model()
        
        # エピソード管理用の変数
        self._current_position: Optional[Tuple[int, int]] = None
    
    def _build_transition_model(self) -> Dict[Tuple[int, int], Dict[str, Dict[Tuple[int, int], float]]]:
        """
        状態遷移確率関数 P(s'|s,a) を構築
        
        Returns:
            遷移確率の辞書構造
            {state: {action: {next_state: probability}}}
        """
        transitions = {}
        
        for state in self._state_space:
            transitions[state] = {}
            for action in self._action_space:
                transitions[state][action] = {}
                
                if self.stochastic:
                    # 確率的遷移：意図した方向に80%、他の方向に各5%
                    intended_next = self._get_next_state(state, action)
                    transitions[state][action][intended_next] = 0.8
                    
                    for other_action in self._action_space:
                        if other_action != action:
                            other_next = self._get_next_state(state, other_action)
                            if other_next in transitions[state][action]:
                                transitions[state][action][other_next] += 0.05
                            else:
                                transitions[state][action][other_next] = 0.05
                    
                    # 確率の正規化（数値誤差を修正）
                    total_prob = sum(transitions[state][action].values())
                    for next_state in transitions[state][action]:
                        transitions[state][action][next_state] /= total_prob
                else:
                    # 決定的遷移
                    next_state = self._get_next_state(state, action)
                    transitions[state][action][next_state] = 1.0
        
        return transitions
    
    def _get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        指定された行動による次の状態を計算（境界処理付き）
        
        Args:
            state: 現在の状態
            action: 実行する行動
            
        Returns:
            次の状態
        """
        i, j = state
        di, dj = self._action_effects[action]
        
        # 境界チェック
        new_i = max(0, min(self.size - 1, i + di))
        new_j = max(0, min(self.size - 1, j + dj))
        
        return (new_i, new_j)
    
    def _build_reward_model(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        報酬関数 R(r|s,a) を構築
        
        Returns:
            報酬の辞書構造 {state: {action: reward}}
        """
        rewards = {}
        
        for state in self._state_space:
            rewards[state] = {}
            for action in self._action_space:
                next_state = self._get_next_state(state, action)
                
                if next_state == self.goal:
                    rewards[state][action] = self.goal_reward
                else:
                    rewards[state][action] = self.move_cost
        
        return rewards
    
    def _sample_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        確率に基づいて次の状態をサンプリング
        
        Args:
            state: 現在の状態
            action: 実行する行動
            
        Returns:
            サンプリングされた次の状態
        """
        probs = self._transition_model[state][action]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # 確率に基づいてサンプリング
        next_state_idx = np.random.choice(len(states), p=probabilities)
        return states[next_state_idx]
    
    # === Environment抽象メソッドの実装 ===
    
    def reset(self, start_position: Optional[Tuple[int, int]] = None, **kwargs) -> Tuple[int, int]:
        """
        環境をリセットして初期状態を返す
        
        Args:
            start_position: 開始位置（Noneの場合はデフォルト位置）
            **kwargs: その他の引数
            
        Returns:
            初期状態の観測
        """
        self._reset_history()
        
        if start_position is not None:
            self._current_position = start_position
        else:
            self._current_position = self.start
        
        return self._current_position
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        行動を実行して次の状態、報酬、終了フラグ、追加情報を返す
        
        Args:
            action: 実行する行動
            
        Returns:
            next_observation: 次の状態の観測
            reward: 受け取った報酬
            done: エピソード終了フラグ
            info: 追加情報を含む辞書
        """
        if self._current_position is None:
            raise ValueError("環境がリセットされていません。reset()を呼び出してください。")
        
        if action not in self._action_space:
            raise ValueError(f"無効な行動: {action}. 有効な行動: {self._action_space}")
        
        current_pos = self._current_position
        
        # 状態遷移の実行（マルコフ性と時不変性を満たす）
        if self.stochastic:
            next_pos = self._sample_next_state(current_pos, action)
        else:
            next_pos = self._get_next_state(current_pos, action)
        
        # 報酬の計算（時不変性：R(r|s,a)は時刻に依存しない）
        reward = self._reward_model[current_pos][action]
        
        # 終了判定
        done = (next_pos == self.goal)
        
        # 状態更新
        self._current_position = next_pos
        
        # 追加情報
        info = {
            "previous_state": current_pos,
            "action": action,
            "deterministic_next": self._get_next_state(current_pos, action),
            "actual_next": next_pos,
            "is_stochastic": self.stochastic
        }
        
        # 履歴に追加
        self._add_to_history(next_pos, action, reward, done, info)
        
        return next_pos, reward, done, info
    
    def get_action_space(self) -> List[str]:
        """利用可能な行動の集合を返す"""
        return self._action_space.copy()
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        環境の現在の状態を可視化する
        
        Args:
            mode: 描画モード ("human", "rgb_array")
            
        Returns:
            mode に応じた描画結果
        """
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"サポートされていない描画モード: {mode}")
    
    def _render_human(self) -> None:
        """人間向けのテキスト描画"""
        print(f"\nグリッドワールド (ステップ: {len(self.get_history())})")
        print(f"現在位置: {self._current_position}")
        print(f"ゴール: {self.goal}")
        
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if (i, j) == self._current_position:
                    row += " A "  # Agent
                elif (i, j) == self.goal:
                    row += " G "  # Goal
                else:
                    row += " . "  # Empty
            print(row)
        print()
    
    def _render_rgb_array(self) -> np.ndarray:
        """RGB配列として描画"""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # グリッドを描画
        grid = np.zeros((self.size, self.size))
        
        # ゴールを設定
        if self.goal:
            grid[self.goal[0], self.goal[1]] = 0.5
        
        # 現在位置を設定
        if self._current_position:
            grid[self._current_position[0], self._current_position[1]] = 1.0
        
        ax.imshow(grid, cmap='RdYlGn', alpha=0.7)
        ax.set_title(f'グリッドワールド (ステップ: {len(self.get_history())})')
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True)
        
        # 軌跡を描画
        if len(self._history) > 0:
            trajectory_x = []
            trajectory_y = []
            for state, _, _ in self._history:
                trajectory_y.append(state[0])
                trajectory_x.append(state[1])
            
            # 現在位置も追加
            if self._current_position:
                trajectory_y.append(self._current_position[0])
                trajectory_x.append(self._current_position[1])
            
            ax.plot(trajectory_x, trajectory_y, 'bo-', alpha=0.6, linewidth=2)
        
        plt.tight_layout()
        
        # シンプルなRGB配列として返す
        plt.close(fig)
        
        # NumPy配列で直接描画
        rgb_array = np.zeros((self.size * 50, self.size * 50, 3), dtype=np.uint8)
        rgb_array.fill(240)  # 薄いグレー背景
        
        cell_size = 50
        for i in range(self.size):
            for j in range(self.size):
                start_y = i * cell_size
                end_y = (i + 1) * cell_size
                start_x = j * cell_size  
                end_x = (j + 1) * cell_size
                
                # ゴールの場合は緑
                if (i, j) == self.goal:
                    rgb_array[start_y:end_y, start_x:end_x] = [0, 255, 0]
                # 現在位置の場合は青
                elif (i, j) == self._current_position:
                    rgb_array[start_y:end_y, start_x:end_x] = [0, 0, 255]
        
        return rgb_array
    
    # === MDPEnvironment抽象メソッドの実装 ===
    
    def get_state_space(self) -> List[Tuple[int, int]]:
        """状態空間を返す"""
        return self._state_space.copy()
    
    def get_transition_model(self) -> Dict[Tuple[int, int], Dict[str, Dict[Tuple[int, int], float]]]:
        """状態遷移モデル P(s'|s,a) を返す"""
        return self._transition_model.copy()
    
    def get_reward_model(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """報酬モデル R(r|s,a) を返す"""
        return self._reward_model.copy()
    
    # === 分析・検証用メソッド ===
    
    def demonstrate_markov_property(self) -> None:
        """マルコフ性の実証"""
        print("=== マルコフ性の実証 ===")
        
        # 異なる履歴から同じ状態に到達するシナリオ
        scenarios = [
            ("シナリオ1", [(0, 0), (0, 1), (1, 1)]),
            ("シナリオ2", [(2, 2), (2, 1), (1, 1)]),
            ("シナリオ3", [(1, 0), (1, 1)])
        ]
        
        target_state = (1, 1)
        test_action = "right"
        
        print(f"目標状態 {target_state} で行動 '{test_action}' を実行:")
        
        for scenario_name, path in scenarios:
            prob_dist = self._transition_model[target_state][test_action]
            print(f"{scenario_name} (履歴: {' → '.join(map(str, path))})")
            print(f"  遷移確率分布: {prob_dist}")
        
        print("→ 履歴に関係なく同じ遷移確率分布を持つ（マルコフ性）")
    
    def demonstrate_time_invariance(self) -> None:
        """時不変性の実証"""
        print("\\n=== 時不変性の実証 ===")
        
        test_state = (1, 1)
        test_action = "right"
        
        print(f"状態 {test_state} で行動 '{test_action}' の遷移確率:")
        prob_dist = self._transition_model[test_state][test_action]
        reward = self._reward_model[test_state][test_action]
        
        print(f"  遷移確率: {prob_dist}")
        print(f"  報酬: {reward}")
        print("この確率と報酬は時刻 t=0, t=100, t=1000 でも同じ値")
        print("→ 時不変性（定常性）")
    
    def get_current_state(self) -> Optional[Tuple[int, int]]:
        """現在の内部状態を返す（デバッグ用）"""
        return self._current_position
    
    def analyze_episode(self) -> Dict[str, Any]:
        """エピソードの分析結果を返す"""
        history = self.get_history()
        if not history:
            return {"total_steps": 0, "total_reward": 0.0, "reached_goal": False}
        
        total_reward = sum(entry["reward"] for entry in history)
        reached_goal = self.is_done()
        
        return {
            "total_steps": len(history),
            "total_reward": total_reward,
            "reached_goal": reached_goal,
            "final_position": self._current_position,
            "trajectory": [entry["observation"] for entry in history]
        }