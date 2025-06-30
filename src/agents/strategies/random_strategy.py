"""
ランダム戦略の実装

ベースライン比較用のランダム行動選択戦略を提供。
"""

from typing import List, Dict, Any
import numpy as np
from ..strategy_interfaces import ModelFreeStrategy
from ..types import Experience, ObservationType, ActionType


class RandomStrategy(ModelFreeStrategy):
    """ランダム行動選択戦略（ベースライン）"""
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """ランダム戦略では全ての行動が均等確率"""
        uniform_prob = 1.0 / len(available_actions)
        return {action: uniform_prob for action in available_actions}
    
    def _observation_to_state(self, observation):
        """観測から状態への変換（ランダム戦略では使用しない）"""
        return observation
    
    def update_policy(self, experience: Experience) -> None:
        """ランダム戦略では学習を行わない"""
        pass