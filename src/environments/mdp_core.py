"""
MDP（マルコフ決定過程）の核となる定義

状態空間、行動空間、遷移モデル、報酬モデルを抽象化し、
環境実装と戦略実装の両方で共有可能な軽量な設計を提供する。
"""

from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Callable
from dataclasses import dataclass

# 型変数の定義
StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')


@dataclass(frozen=True)
class MDPCore(Generic[StateType, ActionType]):
    """
    MDPの核となる情報を集約したデータクラス
    
    環境と戦略の間で共有される軽量なMDP定義。
    実際の環境実装や複雑な状態管理は含まない。
    """
    states: List[StateType]
    actions: List[ActionType]
    transition_model: Callable[[StateType, ActionType, StateType], float]
    reward_model: Callable[[StateType, ActionType, StateType], float]
    observation_to_state: Callable[[any], StateType]
    
    def validate(self) -> bool:
        """
        MDPの整合性をチェックする
        
        Returns:
            整合性がある場合True
            
        Raises:
            ValueError: MDPが整合性を満たさない場合
        """
        # 遷移確率の合計が1になることをチェック
        for state in self.states:
            for action in self.actions:
                total_prob = sum(
                    self.transition_model(state, action, next_state) 
                    for next_state in self.states
                )
                if abs(total_prob - 1.0) > 1e-10:
                    raise ValueError(
                        f"遷移確率の合計が1ではありません: "
                        f"P(·|{state}, {action}) = {total_prob}"
                    )
        
        return True
    
