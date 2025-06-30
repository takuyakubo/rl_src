"""
MDP（マルコフ決定過程）環境の抽象基底クラス

ValueIterationStrategyなどのMDPベースのアルゴリズムで必要な
完全な状態空間、行動空間、遷移モデル、報酬モデルを提供する。
"""

from abc import ABC, abstractmethod
from typing import List, Set, Any, Optional
from .base import Environment, StateType, ActionType, ObservationType
from .mdp_core import MDPCore


class MDPEnvironment(Environment[StateType, ActionType, ObservationType]):
    """
    MDP（マルコフ決定過程）環境の具象クラス
    
    MDPCoreを受け取って、実際の環境シミュレーションを提供する。
    """
    
    def __init__(self, mdp_core: MDPCore[StateType, ActionType], **kwargs):
        """
        MDPCoreから環境を構築
        
        Args:
            mdp_core: MDP定義
            **kwargs: 基底クラスの引数
        """
        super().__init__(**kwargs)
        self.mdp_core = mdp_core
    
    @property
    def states(self) -> List[StateType]:
        """状態空間 S のリストを返す"""
        return self.mdp_core.states
    
    @property
    def actions(self) -> List[ActionType]:
        """行動空間 A のリストを返す"""
        return self.mdp_core.actions
    
    def transition_model(self, state: StateType, action: ActionType, next_state: StateType) -> float:
        """状態遷移確率 P(s_{t+1} | s_t, a_t) を返す"""
        return self.mdp_core.transition_model(state, action, next_state)
    
    def reward_model(self, state: StateType, action: ActionType, next_state: StateType) -> float:
        """報酬関数 R(s_t, a_t, s_{t+1}) を返す"""
        return self.mdp_core.reward_model(state, action, next_state)
    
    def observation_to_state(self, observation: ObservationType) -> StateType:
        """観測から状態への変換"""
        return self.mdp_core.observation_to_state(observation)
    
    
    def validate_mdp(self) -> bool:
        """
        MDPの整合性をチェックする
        
        Returns:
            整合性がある場合True
        """
        return self.mdp_core.validate()
    
    def get_mdp_core(self) -> MDPCore[StateType, ActionType]:
        """
        MDPの核となる情報を返す
        
        Returns:
            MDPCore インスタンス
        """
        return self.mdp_core