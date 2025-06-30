"""
価値反復法戦略の実装

プランニング戦略の具体実装として価値反復法を提供。
"""

from typing import Dict, Any, Optional
from ..strategy_interfaces import PlanningStrategy
from ...environments.mdp_core import MDPCore


class ValueIterationStrategy(PlanningStrategy):
    """価値反復法によるプランニング戦略"""
    
    def __init__(
        self, 
        mdp_core: MDPCore,
        theta: float = 1e-6, 
        max_iterations: int = 1000, 
        gamma: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mdp_core = mdp_core
        self.theta = theta
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.V: Dict[Any, float] = {state: 0.0 for state in self.mdp_core.states}
    
    def _get_reachable_states(self, state, action):
        """
        指定された状態と行動から到達可能な状態のリストを返す（内部最適化用）
        
        Args:
            state: 現在の状態
            action: 実行する行動
            
        Returns:
            到達可能な状態のリスト（確率が0より大きい状態のみ）
        """
        reachable_states = []
        for next_state in self.mdp_core.states:
            if self.mdp_core.transition_model(state, action, next_state) > 0:
                reachable_states.append(next_state)
        return reachable_states
    
    def plan(self) -> Dict[Any, Any]:
        """価値反復法で最適方策を計算"""
        # 価値関数の収束まで反復
        for iteration in range(self.max_iterations):
            delta = 0.0
            
            for state in self.mdp_core.states:
                v_old = self.V[state]
                
                # ベルマン最適方程式を適用
                action_values = []
                for action in self.mdp_core.actions:
                    q_value = 0.0
                    # 最適化：到達可能な状態のみを処理
                    for next_state in self._get_reachable_states(state, action):
                        prob = self.mdp_core.transition_model(state, action, next_state)
                        reward = self.mdp_core.reward_model(state, action, next_state)
                        q_value += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(q_value)
                
                self.V[state] = max(action_values) if action_values else 0.0
                delta = max(delta, abs(v_old - self.V[state]))
            
            if delta < self.theta:
                break
        
        # 貪欲方策を計算
        for state in self.mdp_core.states:
            best_action = None
            best_value = float('-inf')
            
            for action in self.mdp_core.actions:
                q_value = 0.0
                # 最適化：到達可能な状態のみを処理
                for next_state in self._get_reachable_states(state, action):
                    prob = self.mdp_core.transition_model(state, action, next_state)
                    reward = self.mdp_core.reward_model(state, action, next_state)
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            self.policy[state] = best_action
        
        self._is_planned = True
        return self.policy.copy()
    
    def _observation_to_state(self, observation):
        """観測を状態に変換"""
        return self.mdp_core.observation_to_state(observation)