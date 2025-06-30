"""
環境の抽象基底クラス

全ての強化学習環境が実装すべき基本的なインターフェースを定義する。
特定の理論的枠組み（MDPなど）には依存しない汎用的な設計。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar

from .types import StateTransition, NoiseChannel


# 型変数の定義
StateType = TypeVar('StateType')  # 環境の内部状態
ActionType = TypeVar('ActionType')  # 行動
ObservationType = TypeVar('ObservationType')  # 観測 = NoisedState


class Environment(ABC, Generic[StateType, ActionType, ObservationType]):
    """
    強化学習環境の抽象基底クラス
    
    エージェントと環境の相互作用を定義する最小限のインターフェース。
    どの理論的枠組み（MDP、POMDP、マルチエージェントなど）でも利用可能。
    """
    
    def __init__(self, noise_channel: Optional[NoiseChannel[StateType, ObservationType]] = None):
        """
        環境を初期化する
        
        Args:
            noise_channel: ノイズチャネル P(O|S) （dependency injection）
        """
        self._history: List[StateTransition[StateType, ActionType]] = []
        self._current_state: Optional[StateType] = None
        self._current_observation: Optional[ObservationType] = None
        self._episode_done: bool = False
        self._noise_channel = noise_channel
        self._seed: Optional[int] = None
    
    @abstractmethod
    def internal_reset(self, **kwargs) -> StateType:
        """
        環境の内部状態をリセットして初期状態を返す
        
        Args:
            **kwargs: 環境固有のリセット引数
            
        Returns:
            初期内部状態
        """
        pass
    
    def reset(self, **kwargs) -> ObservationType:
        """
        環境をリセットして初期観測を返す
        
        Args:
            **kwargs: 環境固有のリセット引数
            
        Returns:
            初期観測（NoisedState）
        """
        self._reset_history()
        
        # 内部状態をリセット
        initial_state = self.internal_reset(**kwargs)
        self._current_state = initial_state
        
        # 初期観測を生成
        initial_observation = self._generate_observation(initial_state)
        self._current_observation = initial_observation
        
        return initial_observation
    
    @abstractmethod
    def internal_step(self, action: ActionType) -> tuple[StateType, float, bool, Dict[str, Any]]:
        """
        環境内部での純粋な状態遷移を実行する
        
        このメソッドは環境内部での制御を行う：
        1. 行動を受け取り状態遷移を実行
        2. 報酬を計算
        3. 履歴への記録はしない（stepメソッドで行う）
        
        Args:
            action: 実行する行動
            
        Returns:
            next_state: 次の内部状態
            reward: 受け取った報酬
            done: エピソード終了フラグ
            info: 追加情報
        """
        pass
    
    def step(self, action: ActionType) -> tuple[ObservationType, float, bool, Dict[str, Any]]:
        """
        行動を実行して観測を返す（プレイヤー側インターフェース）
        
        このメソッドは以下の手順で動作する：
        1. internal_stepで状態遷移を実行
        2. ノイズチャネルで状態から観測を生成
        3. 履歴に記録
        
        Args:
            action: 実行する行動
            
        Returns:
            observation: 次の観測（NoisedState）
            reward: 受け取った報酬
            done: エピソード終了フラグ
            info: 追加情報
        """
        # 初期化チェック
        if self._current_state is None:
            raise RuntimeError(
                "step() called before reset(). "
                "You must call reset() before the first step()."
            )
        
        previous_state = self._current_state
        
        # 1. 環境内部での状態遷移を実行
        next_state, reward, done, info = self.internal_step(action)
        
        # 2. 観測を生成（各ステップで新規生成）
        observation = self._generate_observation(next_state)
        self._current_observation = observation
        
        # 3. 履歴に記録
        self._add_state_transition(
            previous_state=previous_state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            info=info
        )
        
        return observation, reward, done, info
    
    @abstractmethod
    def get_action_space(self) -> List[ActionType]:
        """
        利用可能な行動の集合を返す
        
        Returns:
            行動集合のリスト
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, state: Optional[StateType] = None) -> List[ActionType]:
        """
        指定された状態で利用可能な行動のリストを返す
        
        Args:
            state: 状態。Noneの場合は現在の状態を使用
            
        Returns:
            利用可能な行動のリスト
        """
        pass
    
    
    def get_history(self) -> List[StateTransition[StateType, ActionType]]:
        """
        環境の履歴を返す
        
        Returns:
            状態遷移の履歴
        """
        return self._history.copy()
    
    
    def is_done(self) -> bool:
        """エピソードが終了しているかを返す"""
        return self._episode_done
    
    def get_noise_channel(self) -> Optional[NoiseChannel[StateType, ObservationType]]:
        """
        ノイズチャネル P(O|S) へのアクセスを提供する
        
        Returns:
            ノイズチャネル、設定されていない場合はNone
        """
        return self._noise_channel
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        環境の乱数シードを設定する
        
        Args:
            seed: 乱数シード
        """
        self._seed = seed
    
    def get_seed(self) -> Optional[int]:
        """
        現在の乱数シードを返す
        
        Returns:
            現在設定されているシード、未設定の場合はNone
        """
        return self._seed
    
    # === 継承クラス用のヘルパーメソッド ===
    
    def _reset_history(self) -> None:
        """履歴をリセットする"""
        self._history = []
        self._current_state = None
        self._current_observation = None
        self._episode_done = False
    
    def _add_state_transition(self, previous_state: StateType, action: ActionType,
                             next_state: StateType, reward: float, done: bool, 
                             info: Dict[str, Any]) -> None:
        """状態遷移を履歴に追加する"""
        transition = StateTransition(
            previous_state=previous_state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            info=info
        )
        self._history.append(transition)
        self._current_state = next_state
        self._episode_done = done
    
    def _generate_observation(self, state: StateType) -> ObservationType:
        """
        状態から観測を生成する
        
        各ステップで新たに観測を生成する。
        同じ状態でも異なるステップでは異なる観測が生成される可能性がある。
        
        Args:
            state: 内部状態
            
        Returns:
            生成された観測
        """
        if self._noise_channel is None:
            raise RuntimeError(
                "NoiseChannel is not set. "
                "You must provide a NoiseChannel when initializing the environment "
                "to generate observations from states."
            )
        
        observation = self._noise_channel.sample_observation(state)
        return observation
    
    def get_current_observation(self) -> Optional[ObservationType]:
        """
        現在の観測を返す
        
        同じステップ内で何度呼ばれても同じ観測を返す。
        
        Returns:
            現在の観測、まだ一度もstepされていない場合はNone
        """
        return self._current_observation
    
    def get_current_state(self) -> Optional[StateType]:
        """
        現在の内部状態を返す
        
        Returns:
            現在の状態、初期化されていない場合はNone
        """
        return self._current_state
    