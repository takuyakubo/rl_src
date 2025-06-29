"""
環境で使用する型定義

Pydanticを使用して、環境の状態遷移の型安全性を提供する。
環境は純粋に状態遷移のみを管理し、観測は外部で処理される。
"""

from typing import Any, Dict, Generic, TypeVar, Protocol
from pydantic import BaseModel, Field

# 型変数の定義
StateType = TypeVar('StateType')  # 環境の内部状態
ActionType = TypeVar('ActionType')  # 行動
ObservationType = TypeVar('ObservationType')  # 観測 = NoisedState


class StateTransition(BaseModel, Generic[StateType, ActionType]):
    """
    環境内部での状態遷移エントリ
    
    環境が内部的に管理する状態遷移の記録。観測の記録はエージェント側の責務。
    """
    previous_state: StateType = Field(description="遷移前の内部状態")
    action: ActionType = Field(description="実行された行動")
    next_state: StateType = Field(description="遷移後の内部状態")
    reward: float = Field(description="受け取った報酬")
    done: bool = Field(description="エピソード終了フラグ")
    info: Dict[str, Any] = Field(default_factory=dict, description="追加情報")

    class Config:
        arbitrary_types_allowed = True


class NoiseChannel(Protocol, Generic[StateType, ObservationType]):
    """
    ノイズチャネルのプロトコル
    
    状態から観測への条件付き確率 P(O|S) を実装する。
    NoisedState = Observation として、環境はこのチャネルを通したObservationを返す。
    """
    
    def sample_observation(self, state: StateType) -> ObservationType:
        """
        与えられた状態から観測をサンプリングする
        
        Args:
            state: 現在の内部状態
            
        Returns:
            サンプリングされた観測（NoisedState）
        """
        ...
    
    def get_observation_probability(self, observation: ObservationType, state: StateType) -> float:
        """
        P(O|S) の確率を返す
        
        Args:
            observation: 観測
            state: 状態
            
        Returns:
            条件付き確率 P(observation|state)
        """
        ...


