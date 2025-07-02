"""
TD学習戦略の実装

モデルフリー戦略の具体実装としてTD(0)学習を提供。
価値関数の学習にTD誤差を使用し、ε-貪欲方策で行動選択を行う。
"""

from typing import Dict, Any, List, Optional
import numpy as np
from ..strategy_interfaces import ModelFreeStrategy
from ..types import Experience, ObservationType, ActionType


class TDZeroStrategy(ModelFreeStrategy):
    """TD(0)学習戦略"""
    
    def __init__(
        self, 
        epsilon: float = 0.1, 
        learning_rate: float = 0.1, 
        gamma: float = 0.9,
        initial_value: float = 0.0,
        **kwargs
    ):
        """
        TD(0)学習戦略を初期化
        
        Args:
            epsilon: ε-貪欲方策の探索率
            learning_rate: 学習率（α）
            gamma: 割引率
            initial_value: 価値関数の初期値
        """
        super().__init__(gamma=gamma, **kwargs)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        
        # 状態価値関数 V(s)
        self.V: Dict[Any, float] = {}
        
        # Q値を近似的に計算するための報酬モデル（経験から推定）
        self.reward_estimates: Dict[Any, Dict[Any, List[float]]] = {}
        
        # 遷移確率の推定（経験から推定）
        self.transition_counts: Dict[Any, Dict[Any, Dict[Any, int]]] = {}
        
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """現在の観測での行動分布を取得（ε-貪欲方策）"""
        state = self._observation_to_state(observation)
        
        # 各行動のQ値を推定
        q_values = {}
        for action in available_actions:
            q_values[action] = self._estimate_q_value(state, action)
        
        # 最適行動を決定
        if q_values:
            best_action = max(q_values.keys(), key=lambda a: q_values[a])
        else:
            # Q値がない場合はランダムに選択
            best_action = np.random.choice(available_actions)
        
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
    
    def _estimate_q_value(self, state: Any, action: Any) -> float:
        """
        Q値を推定（V値と経験から）
        Q(s,a) ≈ r(s,a) + γ * Σ P(s'|s,a) * V(s')
        """
        # 報酬の推定値
        if state in self.reward_estimates and action in self.reward_estimates[state]:
            rewards = self.reward_estimates[state][action]
            expected_reward = np.mean(rewards) if rewards else 0.0
        else:
            expected_reward = 0.0
        
        # 次状態の価値の期待値
        expected_next_value = 0.0
        if state in self.transition_counts and action in self.transition_counts[state]:
            total_transitions = sum(self.transition_counts[state][action].values())
            if total_transitions > 0:
                for next_state, count in self.transition_counts[state][action].items():
                    prob = count / total_transitions
                    next_value = self.V.get(next_state, self.initial_value)
                    expected_next_value += prob * next_value
        
        return expected_reward + self.gamma * expected_next_value
    
    def _observation_to_state(self, observation):
        """観測から状態への変換（デフォルトでは観測をそのまま状態とする）"""
        return observation
    
    def update_policy(self, experience: Experience) -> None:
        """TD(0)アルゴリズムで価値関数を更新"""
        state = self._observation_to_state(experience.observation)
        next_state = self._observation_to_state(experience.next_observation)
        
        # 価値関数を初期化
        if state not in self.V:
            self.V[state] = self.initial_value
        if next_state not in self.V:
            self.V[next_state] = self.initial_value
        
        # TD誤差を計算
        if experience.done:
            # 終端状態の場合
            td_error = experience.reward - self.V[state]
        else:
            # 通常の遷移
            td_error = experience.reward + self.gamma * self.V[next_state] - self.V[state]
        
        # 価値関数を更新
        self.V[state] += self.learning_rate * td_error
        
        # 報酬モデルを更新
        if state not in self.reward_estimates:
            self.reward_estimates[state] = {}
        if experience.action not in self.reward_estimates[state]:
            self.reward_estimates[state][experience.action] = []
        self.reward_estimates[state][experience.action].append(experience.reward)
        
        # 遷移モデルを更新
        if not experience.done:
            if state not in self.transition_counts:
                self.transition_counts[state] = {}
            if experience.action not in self.transition_counts[state]:
                self.transition_counts[state][experience.action] = {}
            if next_state not in self.transition_counts[state][experience.action]:
                self.transition_counts[state][experience.action][next_state] = 0
            self.transition_counts[state][experience.action][next_state] += 1
    
    def reset(self) -> None:
        """エピソード開始時のリセット（TD(0)では特に処理なし）"""
        pass
    
    def get_value_function(self) -> Dict[Any, float]:
        """学習した価値関数を取得（デバッグ・可視化用）"""
        return self.V.copy()
    
    def get_td_error_info(self, experience: Experience) -> Dict[str, float]:
        """TD誤差の詳細情報を取得（デバッグ・可視化用）"""
        state = self._observation_to_state(experience.observation)
        next_state = self._observation_to_state(experience.next_observation)
        
        current_value = self.V.get(state, self.initial_value)
        next_value = self.V.get(next_state, self.initial_value) if not experience.done else 0.0
        
        if experience.done:
            td_error = experience.reward - current_value
        else:
            td_error = experience.reward + self.gamma * next_value - current_value
        
        return {
            "current_value": current_value,
            "next_value": next_value,
            "reward": experience.reward,
            "td_error": td_error,
            "done": experience.done
        }