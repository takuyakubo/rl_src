"""
Q学習戦略の実装

モデルフリー戦略の具体実装としてε-貪欲なQ学習を提供。
"""

from typing import Dict, Any, List
import numpy as np
from ..strategy_interfaces import ModelFreeStrategy
from ..types import Experience, ObservationType, ActionType


class EpsilonGreedyStrategy(ModelFreeStrategy):
    """ε-貪欲戦略（Q学習などの基盤）"""
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.1, gamma: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q: Dict[Any, Dict[Any, float]] = {}
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """現在の観測での行動分布を取得（ε-貪欲方策）"""
        state = self._observation_to_state(observation)
        
        # Q値から最適行動を決定
        best_action = None
        best_value = float('-inf')
        
        if state in self.Q:
            for action in available_actions:
                q_value = self.Q[state].get(action, 0.0)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
        
        if best_action is None:
            # Q値がない場合は均等分布
            uniform_prob = 1.0 / len(available_actions)
            return {action: uniform_prob for action in available_actions}
        
        # ε-貪欲方策の確率分布
        action_probs = {}
        explore_prob = self.epsilon / len(available_actions)
        
        for action in available_actions:
            if action == best_action:
                # 最適行動：(1-ε) + ε/|A|
                action_probs[action] = (1.0 - self.epsilon) + explore_prob
            else:
                # その他の行動：ε/|A|
                action_probs[action] = explore_prob
        
        return action_probs
    
    def _observation_to_state(self, observation):
        """観測から状態への変換（デフォルトでは観測をそのまま状態とする）"""
        return observation
    
    def update_policy(self, experience: Experience) -> None:
        """Q値を更新"""
        state = self._observation_to_state(experience.observation)
        next_state = self._observation_to_state(experience.next_observation)
        
        # Q-tableを初期化
        if state not in self.Q:
            self.Q[state] = {}
        if experience.action not in self.Q[state]:
            self.Q[state][experience.action] = 0.0
        
        # 次状態の最大Q値を計算
        max_next_q = 0.0
        if not experience.done and next_state in self.Q:
            max_next_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
        
        # Q値更新（Q学習の更新式）
        current_q = self.Q[state][experience.action]
        target_q = experience.reward + self.gamma * max_next_q
        self.Q[state][experience.action] = current_q + self.learning_rate * (target_q - current_q)