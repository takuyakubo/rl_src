"""
方策決定戦略のインターフェース

Planning, Model-based, Model-freeの3つのアプローチを
統一的なインターフェースで抽象化し、Dependency Injectionを可能にする。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Protocol, Generic
from .types import Experience, ObservationType, ActionType


class PolicyStrategy(Protocol[ObservationType, ActionType]):
    """
    方策決定戦略のプロトコル
    
    すべての方策決定戦略が実装すべき統一インターフェース。
    エージェント作成時に具体的な戦略を注入する。
    """
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """
        現在の観測での行動分布を取得する
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
        
        Returns:
            行動分布（{行動: 確率} の辞書）
        """
        ...
    
    def get_action(self, observation: ObservationType, available_actions: List[ActionType]) -> ActionType:
        """
        観測に基づいて行動を決定する
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
            
        Returns:
            選択された行動
        """
        ...
    
    def update(self, experience: Experience) -> None:
        """
        経験から学習する
        
        Args:
            experience: 観測された経験
        """
        ...
    
    def reset(self) -> None:
        """
        エピソード開始時のリセット
        """
        ...


class PlanningStrategy(PolicyStrategy[ObservationType, ActionType], ABC):
    """
    プランニング戦略の抽象基底クラス
    
    事前に環境モデルが既知で、最適方策を事前計算する戦略。
    MDPEnvironmentを通じて完全な環境情報にアクセスする。
    """
    
    def __init__(self):
        """
        プランニング戦略を初期化
        
        具体的な戦略クラスでMDPEnvironmentを受け取る
        """
        # 計算された方策
        self.policy: Dict[Any, Any] = {}
        self._is_planned = False
    
    @abstractmethod
    def plan(self) -> Dict[Any, Any]:
        """
        最適方策を事前計算する
        
        Returns:
            計算された方策（状態 -> 行動の辞書）
        """
        pass
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """現在の観測での行動分布を取得（確率>0の行動のみ）"""
        if not self._is_planned:
            self.policy = self.plan()
            self._is_planned = True
        
        # 観測から状態を取得
        state = self._observation_to_state(observation)
        
        # 該当状態の行動を取得
        if state in self.policy:
            best_action = self.policy[state]
            if best_action in available_actions:
                # 決定的方策：最適行動のみに確率1
                return {best_action: 1.0}
        
        # 方策にない場合は全ての行動が均等（確率>0なので全て含む）
        uniform_prob = 1.0 / len(available_actions)
        return {action: uniform_prob for action in available_actions}
    
    def get_action(self, observation: ObservationType, available_actions: List[ActionType]) -> ActionType:
        """観測に基づいて行動を決定（policyを使用）"""
        action_probs = self.get_policy(observation, available_actions)
        
        # 確率に基づいてサンプリング
        import numpy as np
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    @abstractmethod
    def _observation_to_state(self, observation: ObservationType) -> Any:
        """観測を状態に変換（サブクラスで実装）"""
        pass
    
    def update(self, experience: Experience) -> None:
        """プランニング戦略では基本的にオンライン学習を行わない"""
        pass
    
    def reset(self) -> None:
        """リセット時の処理"""
        pass
    
    def is_planned(self) -> bool:
        """方策が計算済みかどうか"""
        return self._is_planned


class ModelBasedStrategy(PolicyStrategy[ObservationType, ActionType], ABC):
    """
    モデルベース戦略の抽象基底クラス
    
    環境モデルを学習し、そのモデルを使って方策を改善する戦略。
    """
    
    def __init__(
        self,
        states: List[Any],
        actions: List[ActionType],
        observation_to_state: Callable[[ObservationType], Any],
        gamma: float = 0.9
    ):
        """
        モデルベース戦略を初期化
        
        Args:
            states: 状態空間
            actions: 行動空間  
            observation_to_state: 観測を状態に変換する関数
            gamma: 割引率
        """
        self.states = states
        self.actions = actions
        self.observation_to_state = observation_to_state
        self.gamma = gamma
        
        # 学習される環境モデル
        self.transition_counts: Dict[Any, Dict[Any, Dict[Any, int]]] = {}
        self.reward_sums: Dict[Any, Dict[Any, float]] = {}
        self.reward_counts: Dict[Any, Dict[Any, int]] = {}
        
        # 方策
        self.policy: Dict[Any, Any] = {}
        
        # 学習統計
        self.model_updates = 0
        self.policy_updates = 0
    
    @abstractmethod
    def update_model(self, experience: Experience) -> None:
        """
        経験から環境モデルを更新
        
        Args:
            experience: 観測された経験
        """
        pass
    
    @abstractmethod
    def update_policy(self) -> None:
        """
        現在の環境モデルを使って方策を更新
        """
        pass
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """現在の観測での行動分布を取得（確率>0の行動のみ）"""
        state = self.observation_to_state(observation)
        
        # 該当状態の行動を取得
        if state in self.policy:
            best_action = self.policy[state]
            if best_action in available_actions:
                # 決定的方策：最適行動のみに確率1
                return {best_action: 1.0}
        
        # 方策にない場合は全ての行動が均等（確率>0なので全て含む）
        uniform_prob = 1.0 / len(available_actions)
        return {action: uniform_prob for action in available_actions}
    
    def get_action(self, observation: ObservationType, available_actions: List[ActionType]) -> ActionType:
        """観測に基づいて行動を決定（policyを使用）"""
        action_probs = self.get_policy(observation, available_actions)
        
        # 確率に基づいてサンプリング
        import numpy as np
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def update(self, experience: Experience) -> None:
        """経験からモデルと方策を更新"""
        # 1. 環境モデルを更新
        self.update_model(experience)
        self.model_updates += 1
        
        # 2. 方策を更新（条件に応じて）
        if self._should_update_policy():
            self.update_policy()
            self.policy_updates += 1
    
    @abstractmethod
    def _should_update_policy(self) -> bool:
        """方策を更新すべきかどうかを判定"""
        pass
    
    def reset(self) -> None:
        """リセット時の処理"""
        pass


class ModelFreeStrategy(PolicyStrategy[ObservationType, ActionType], ABC):
    """
    モデルフリー戦略の抽象基底クラス
    
    環境モデルを明示的に学習せず、直接価値関数や方策を学習する戦略。
    """
    
    def __init__(self, gamma: float = 0.9, **kwargs):
        """
        モデルフリー戦略を初期化
        
        Args:
            gamma: 割引率
        """
        self.gamma = gamma
        
        # 学習統計
        self.learning_updates = 0
    
    @abstractmethod
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """
        現在の観測での行動分布を取得する（確率>0の行動のみ）
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
        
        Returns:
            行動分布（{行動: 確率} の辞書）
            
        Note:
            Model-freeでは価値関数から方策を導出する場合が多い
        """
        pass
    
    @abstractmethod
    def update_policy(self, experience: Experience) -> None:
        """
        経験から直接方策を更新
        
        Args:
            experience: 観測された経験
        """
        pass
    
    def get_action(self, observation: ObservationType, available_actions: List[ActionType]) -> ActionType:
        """
        観測に基づいて行動を決定（policyを使用）
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
            
        Returns:
            選択された行動
        """
        action_probs = self.get_policy(observation, available_actions)
        
        # 確率に基づいてサンプリング
        import numpy as np
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    @abstractmethod
    def _observation_to_state(self, observation: ObservationType) -> Any:
        """観測を状態に変換（サブクラスで実装）"""
        pass
    
    def update(self, experience: Experience) -> None:
        """経験から学習"""
        self.update_policy(experience)
        self.learning_updates += 1
    
    def reset(self) -> None:
        """リセット時の処理"""
        pass
