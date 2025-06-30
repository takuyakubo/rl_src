"""
GridWorld MDPCore ファクトリー

GridWorldのMDPCoreを構築するためのファクトリー関数。
MDPCoreを中心とした設計により、環境と戦略が同じデータを共有する。
"""

import numpy as np
from typing import Tuple, List, Optional
from src.environments.mdp_core import MDPCore


def create_gridworld_mdp_core(
    size: int = 3,
    goal: Tuple[int, int] = (2, 2),
    stochastic: bool = False,
    move_cost: float = -0.1,
    goal_reward: float = 1.0,
    random_seed: Optional[int] = None
) -> MDPCore[Tuple[int, int], str]:
    """
    GridWorld MDPCoreを作成
    
    Args:
        size: グリッドのサイズ (size × size)
        goal: ゴール位置の座標
        stochastic: 確率的遷移を使用するか
        move_cost: 移動時のコスト（負の値）
        goal_reward: ゴール到達時の報酬
        random_seed: 乱数シード
        
    Returns:
        GridWorld用のMDPCore
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 状態空間 S
    states = [(i, j) for i in range(size) for j in range(size)]
    
    # 行動空間 A
    actions = ["up", "right", "down", "left"]
    action_effects = {
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1)
    }
    
    def get_next_state(state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """境界処理付きの次状態計算"""
        i, j = state
        di, dj = action_effects[action]
        new_i = max(0, min(size - 1, i + di))
        new_j = max(0, min(size - 1, j + dj))
        return (new_i, new_j)
    
    def transition_model(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        """状態遷移確率 P(s_{t+1} | s_t, a_t)"""
        if not stochastic:
            # 決定的環境
            intended_next = get_next_state(state, action)
            return 1.0 if next_state == intended_next else 0.0
        else:
            # 確率的環境：意図した方向80%、他の方向各5%
            w_intended = 0.8
            w_other = 0.05
            
            # 全行動について次状態を計算し、重みを割り当て
            next_states = [get_next_state(state, a) for a in actions]
            weights = np.array([w_intended if a == action else w_other for a in actions])
            probabilities = weights / weights.sum()
            
            # 指定された次状態に対応する確率を返す
            try:
                state_index = next_states.index(next_state)
                return probabilities[state_index]
            except ValueError:
                return 0.0
    
    def reward_model(state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        """報酬関数 R(s_t, a_t, s_{t+1})"""
        return goal_reward if next_state == goal else move_cost
    
    def observation_to_state(observation: Tuple[int, int]) -> Tuple[int, int]:
        """観測から状態への変換（GridWorldでは恒等関数）"""
        return observation
    
    return MDPCore(
        states=states,
        actions=actions,
        transition_model=transition_model,
        reward_model=reward_model,
        observation_to_state=observation_to_state
    )