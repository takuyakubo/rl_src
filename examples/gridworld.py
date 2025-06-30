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
from environments.mdp import MDPEnvironment
from environments.mdp_core import MDPCore
from environments.types import NoiseChannel


class DeterministicGridWorldChannel(NoiseChannel[Tuple[int, int], Tuple[int, int]]):
    """決定的なグリッドワールド用ノイズチャネル"""
    
    def sample_observation(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """状態をそのまま観測として返す"""
        return state
    
    def get_observation_probability(self, observation: Tuple[int, int], state: Tuple[int, int]) -> float:
        """決定的なので、状態と観測が一致する場合のみ1.0"""
        return 1.0 if observation == state else 0.0


class GridWorldEnvironment(MDPEnvironment[Tuple[int, int], str, Tuple[int, int]]):
    """
    グリッドワールド環境
    
    MDPCoreを受け取って、実際のグリッドワールドシミュレーションを提供する。
    """
    
    def __init__(
        self,
        mdp_core: MDPCore[Tuple[int, int], str],
        goal: Tuple[int, int] = (2, 2),
        start: Optional[Tuple[int, int]] = None,
        random_seed: Optional[int] = None
    ):
        """
        グリッドワールドを初期化
        
        Args:
            mdp_core: GridWorld用のMDPCore
            goal: ゴール位置の座標
            start: 開始位置（Noneの場合は(0,0)）
            random_seed: 乱数シード
        """
        # MDPCoreを使ってMDPEnvironmentを初期化
        super().__init__(
            mdp_core=mdp_core,
            noise_channel=DeterministicGridWorldChannel()
        )
        
        # MDPCoreからサイズを導出
        import math
        self.size = int(math.sqrt(len(mdp_core.states)))
        self.goal = goal
        self.start = start if start is not None else (0, 0)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # エピソード管理用の変数
        self._current_position: Optional[Tuple[int, int]] = None
    
    
    def _sample_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        現在の状態と行動から次の状態をサンプリング
        
        Args:
            state: 現在の状態
            action: 実行する行動
            
        Returns:
            サンプリングされた次の状態
        """
        # 全ての可能な遷移先を取得
        possible_states = []
        probabilities = []
        
        for next_state in self.mdp_core.states:
            prob = self.mdp_core.transition_model(state, action, next_state)
            if prob > 0:
                possible_states.append(next_state)
                probabilities.append(prob)
        
        if not possible_states:
            return state  # フォールバック
        
        # 確率に基づいてサンプリング
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()  # 正規化
        chosen_index = np.random.choice(len(possible_states), p=probabilities)
        return possible_states[chosen_index]
    
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
        
        if action not in self.mdp_core.actions:
            raise ValueError(f"無効な行動: {action}. 有効な行動: {self.mdp_core.actions}")
        
        current_pos = self._current_position
        
        # 状態遷移の実行（マルコフ性と時不変性を満たす）
        next_pos = self._sample_next_state(current_pos, action)
        
        # 報酬の計算（時不変性：R(s,a,s')は時刻に依存しない）
        reward = self.mdp_core.reward_model(current_pos, action, next_pos)
        
        # 終了判定
        done = (next_pos == self.goal)
        
        # 状態更新
        self._current_position = next_pos
        
        # 追加情報
        info = {
            "previous_state": current_pos,
            "action": action,
            "actual_next": next_pos,
        }
        
        return next_pos, reward, done, info
    
    @override
    def get_action_space(self) -> List[str]:
        """利用可能な行動の集合を返す"""
        return self.mdp_core.actions.copy()
    
    @override
    def get_available_actions(self, state: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        指定された状態で利用可能な行動のリストを返す
        
        Args:
            state: 状態。Noneの場合は現在の状態を使用
            
        Returns:
            利用可能な行動のリスト
            
        Note:
            グリッドワールドでは基本的に全ての行動が利用可能だが、
            将来的には壁や障害物による制約を実装可能
        """
        if state is None:
            if self._current_position is None:
                raise ValueError("環境がリセットされていません")
            state = self._current_position
        
        # グリッドワールドでは現在全ての行動が利用可能
        # 将来的には壁や障害物による制約を実装可能
        available_actions = self.mdp_core.actions.copy()
        
        # 例：境界での制約を実装する場合（現在はコメントアウト）
        # i, j = state
        # if i == 0:  # 上端
        #     available_actions = [a for a in available_actions if a != "up"]
        # if i == self.size - 1:  # 下端
        #     available_actions = [a for a in available_actions if a != "down"]
        # if j == 0:  # 左端
        #     available_actions = [a for a in available_actions if a != "left"]
        # if j == self.size - 1:  # 右端
        #     available_actions = [a for a in available_actions if a != "right"]
        
        return available_actions