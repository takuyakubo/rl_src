"""
グリッドワールド環境の実装

マルコフ決定過程の具体例として、シンプルなグリッドワールドを実装。
Zenn記事での説明と同じ仕様で、実際に動作するコードを提供する。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any, override
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.base import Environment
from environments.types import NoiseChannel


class DeterministicGridWorldChannel(NoiseChannel[Tuple[int, int], Tuple[int, int]]):
    """決定的なグリッドワールド用ノイズチャネル"""
    
    def sample_observation(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """状態をそのまま観測として返す"""
        return state
    
    def get_observation_probability(self, observation: Tuple[int, int], state: Tuple[int, int]) -> float:
        """決定的なので、状態と観測が一致する場合のみ1.0"""
        return 1.0 if observation == state else 0.0


class GridWorldEnvironment(Environment[Tuple[int, int], str, Tuple[int, int]]):
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
        # 決定的なノイズチャネルを使用
        super().__init__(noise_channel=DeterministicGridWorldChannel())
        
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
        
        
        
        # エピソード管理用の変数
        self._current_position: Optional[Tuple[int, int]] = None
    
    
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
    
    def reward_model(self, state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        """
        報酬関数 R(s_t, a_t, s_{t+1})
        
        Args:
            state: 現在の状態 s_t
            action: 実行した行動 a_t
            next_state: 次の状態 s_{t+1}
            
        Returns:
            報酬値
        """
        return self.goal_reward if next_state == self.goal else self.move_cost
    
    def _sample_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        現在の状態と行動から次の状態をサンプリング
        
        Args:
            state: 現在の状態
            action: 実行する行動
            
        Returns:
            サンプリングされた次の状態
        """
        if not self.stochastic:
            return self._get_next_state(state, action)
        
        # 確率的遷移：意図した方向80%、他の方向各5%
        w_intended = 0.8
        w_other = 0.05
        
        # 全行動について次状態を計算し、重みを割り当て
        actions = self._action_space
        next_states = [self._get_next_state(state, a) for a in actions]
        weights = np.array([w_intended if a == action else w_other for a in actions])
        probabilities = weights / weights.sum()
        
        # インデックスをサンプリングして対応する状態を返す
        chosen_index = np.random.choice(len(next_states), p=probabilities)
        return next_states[chosen_index]
    
    # === Environment抽象メソッドの実装 ===
    
    @override
    def internal_reset(self, start_position: Optional[Tuple[int, int]] = None, **kwargs) -> Tuple[int, int]:
        """
        環境の内部状態をリセットして初期状態を返す
        
        Args:
            start_position: 開始位置（Noneの場合はデフォルト位置）
            **kwargs: その他の引数
            
        Returns:
            初期内部状態
        """
        
        if start_position is not None:
            self._current_position = start_position
        else:
            self._current_position = self.start
        
        return self._current_position
    
    @override
    def internal_step(self, action: str) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        環境内部での純粋な状態遷移を実行する
        
        Args:
            action: 実行する行動
            
        Returns:
            next_state: 次の内部状態
            reward: 受け取った報酬
            done: エピソード終了フラグ
            info: 追加情報を含む辞書
        """
        
        if action not in self._action_space:
            raise ValueError(f"無効な行動: {action}. 有効な行動: {self._action_space}")
        
        current_pos = self._current_position
        
        # 状態遷移の実行（マルコフ性と時不変性を満たす）
        next_pos = self._sample_next_state(current_pos, action)
        
        # 報酬の計算（時不変性：R(s,a,s')は時刻に依存しない）
        reward = self.reward_model(current_pos, action, next_pos)
        
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
        
        return next_pos, reward, done, info
    
    @override
    def get_action_space(self) -> List[str]:
        """利用可能な行動の集合を返す"""
        return self._action_space.copy()