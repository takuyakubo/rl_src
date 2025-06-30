"""
エージェントの基底クラス（改良版）

観測ベースで動作し、方策決定戦略をDIで注入できる統一的な設計。
"""

from abc import abstractmethod
from typing import TypeVar, Generic, Dict, Any, Optional, List, Protocol
import numpy as np

from .types import Experience, ObservationType, ActionType
from .strategy_interfaces import PolicyStrategy



class Agent(Generic[ObservationType, ActionType]):
    """
    強化学習エージェントの基底クラス
    
    観測ベースで動作し、方策決定戦略をDIで注入する統一的な設計。
    すべてのエージェントはこのクラスのインスタンスとして作成される。
    """
    
    def __init__(
        self,
        strategy: PolicyStrategy[ObservationType, ActionType],
        name: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        エージェントを初期化する
        
        Args:
            strategy: 方策決定戦略（DI）
            name: エージェントの名前
            random_seed: 乱数シード
        """
        self.strategy = strategy
        self.name = name or f"Agent({type(strategy).__name__})"
        self._seed = random_seed
        
        # 現在のエピソードの経験追跡
        self._current_episode_experiences: List[Experience] = []
        self._last_observation: Optional[ObservationType] = None
        self._last_action: Optional[ActionType] = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 戦略固有の初期化
        self._initialize_strategy()
    
    def _initialize_strategy(self) -> None:
        """戦略固有の初期化処理"""
        # プランニング戦略の場合は事前計算を実行
        if hasattr(self.strategy, 'plan') and hasattr(self.strategy, 'is_planned'):
            if not self.strategy.is_planned():
                print(f"{self.name}: プランニング戦略の事前計算を実行中...")
                self.strategy.plan()
                print(f"{self.name}: プランニング完了")
    
    def get_action(self, observation: ObservationType, available_actions: List[ActionType]) -> ActionType:
        """
        現在の観測に対する行動を決定する
        
        Args:
            observation: 現在の観測
            available_actions: 現在利用可能な行動のリスト
            
        Returns:
            選択された行動
        """
        return self.strategy.get_action(observation, available_actions)
    
    def step(self, observation: ObservationType, available_actions: List[ActionType], 
             reward: float = 0.0, done: bool = False, 
             info: Optional[Dict[str, Any]] = None) -> ActionType:
        """
        観測を受け取り、経験を記録し、次の行動を決定する
        
        Args:
            observation: 現在の観測
            available_actions: 現在利用可能な行動のリスト
            reward: 前回行動の報酬（初回は0）
            done: エピソード終了フラグ
            info: 追加情報
            
        Returns:
            次に実行する行動
        """
        info = info or {}
        
        # 前回の経験を記録し学習（初回以外）
        if self._last_observation is not None and self._last_action is not None:
            experience = Experience(
                observation=self._last_observation,
                action=self._last_action,
                reward=reward,
                next_observation=observation,
                done=done,
                info=info
            )
            
            # 現在のエピソードの経験に記録
            self._current_episode_experiences.append(experience)
            
            # 戦略の更新
            self.strategy.update(experience)
        
        # エピソード終了時の処理
        if done:
            self._last_observation = None
            self._last_action = None
            return available_actions[0] if available_actions else None  # ダミー行動
        
        # 次の行動を決定
        action = self.get_action(observation, available_actions)
        
        # 状態を更新
        self._last_observation = observation
        self._last_action = action
        
        return action
    
    def reset(self) -> List[Experience]:
        """
        エピソード開始時のリセット
        
        Returns:
            前回のエピソードの経験リスト
        """
        # 前回のエピソードの経験を返す
        episode_experiences = self._current_episode_experiences.copy()
        
        # 現在のエピソードをリセット
        self._current_episode_experiences = []
        self._last_observation = None
        self._last_action = None
        
        # 戦略のリセット
        self.strategy.reset()
        
        return episode_experiences
    
    def get_current_episode_experiences(self) -> List[Experience]:
        """
        現在のエピソードの経験リストを取得する
        
        Returns:
            現在のエピソードの経験リスト
        """
        return self._current_episode_experiences.copy()
    


