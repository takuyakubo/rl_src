"""
エピソード生成ユーティリティ

モンテカルロ法で使用するエピソード生成とリターン計算を提供する。
"""

from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .types import Experience, ActionType, ObservationType
from ..environments.base import Environment


@dataclass
class EpisodeStep:
    """エピソード内の1ステップの情報"""
    state: Any
    action: ActionType
    reward: float
    next_state: Any
    done: bool
    info: Dict[str, Any]


@dataclass
class Episode:
    """完全なエピソードの情報"""
    steps: List[EpisodeStep]
    total_reward: float
    length: int
    
    def get_returns(self, gamma: float = 1.0) -> List[float]:
        """
        各ステップからのリターン（累積割引報酬）を計算
        
        Args:
            gamma: 割引率
            
        Returns:
            各ステップからのリターンのリスト
        """
        returns = []
        G = 0.0
        
        # 逆順にリターンを計算
        for step in reversed(self.steps):
            G = step.reward + gamma * G
            returns.append(G)
        
        # 元の順序に戻す
        returns.reverse()
        return returns
    
    def get_state_action_returns(self, gamma: float = 1.0) -> List[Tuple[Any, ActionType, float]]:
        """
        (状態, 行動, リターン) のタプルのリストを取得
        
        Args:
            gamma: 割引率
            
        Returns:
            (state, action, return) のタプルのリスト
        """
        returns = self.get_returns(gamma)
        return [(step.state, step.action, G) for step, G in zip(self.steps, returns)]


class EpisodeGenerator:
    """
    方策に従ってエピソードを生成するクラス
    """
    
    def __init__(
        self, 
        environment: Environment,
        max_steps: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        エピソード生成器を初期化
        
        Args:
            environment: 環境
            max_steps: 最大ステップ数
            random_seed: 乱数シード
        """
        self.environment = environment
        self.max_steps = max_steps
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_episode(
        self, 
        policy_func: Callable[[Any, List[ActionType]], ActionType],
        start_state: Optional[Any] = None
    ) -> Episode:
        """
        指定された方策に従ってエピソードを生成
        
        Args:
            policy_func: 方策関数 (state, available_actions) -> action
            start_state: 開始状態（Noneの場合はランダム）
            
        Returns:
            生成されたエピソード
        """
        steps = []
        total_reward = 0.0
        
        # エピソード開始
        if start_state is not None:
            current_state = self.environment.reset(start_position=start_state)
        else:
            current_state = self.environment.reset()
        
        for step_count in range(self.max_steps):
            # 利用可能な行動を取得
            available_actions = self.environment.get_available_actions()
            
            # 方策に従って行動を選択
            action = policy_func(current_state, available_actions)
            
            # 行動を実行
            next_state, reward, done, info = self.environment.step(action)
            
            # ステップ情報を記録
            step = EpisodeStep(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info
            )
            steps.append(step)
            total_reward += reward
            
            # 終了判定
            if done:
                break
            
            current_state = next_state
        
        return Episode(
            steps=steps,
            total_reward=total_reward,
            length=len(steps)
        )
    
    def generate_episodes(
        self,
        policy_func: Callable[[Any, List[ActionType]], ActionType],
        num_episodes: int,
        start_states: Optional[List[Any]] = None
    ) -> List[Episode]:
        """
        複数のエピソードを生成
        
        Args:
            policy_func: 方策関数
            num_episodes: 生成するエピソード数
            start_states: 開始状態のリスト（Noneの場合は全てランダム）
            
        Returns:
            生成されたエピソードのリスト
        """
        episodes = []
        
        for i in range(num_episodes):
            start_state = None
            if start_states is not None and i < len(start_states):
                start_state = start_states[i]
            
            episode = self.generate_episode(policy_func, start_state)
            episodes.append(episode)
        
        return episodes


class MonteCarloUtils:
    """モンテカルロ法で使用する共通ユーティリティ"""
    
    @staticmethod
    def first_visit_returns(
        episodes: List[Episode], 
        gamma: float = 1.0
    ) -> Dict[Tuple[Any, ActionType], List[float]]:
        """
        First-visit方式で状態行動ペアのリターンを計算
        
        Args:
            episodes: エピソードのリスト
            gamma: 割引率
            
        Returns:
            {(state, action): [returns]} の辞書
        """
        state_action_returns = {}
        
        for episode in episodes:
            visited = set()
            
            for state, action, G in episode.get_state_action_returns(gamma):
                state_action_pair = (state, action)
                
                # First-visit: この状態行動ペアが初回訪問の場合のみ
                if state_action_pair not in visited:
                    visited.add(state_action_pair)
                    
                    if state_action_pair not in state_action_returns:
                        state_action_returns[state_action_pair] = []
                    
                    state_action_returns[state_action_pair].append(G)
        
        return state_action_returns
    
    @staticmethod
    def every_visit_returns(
        episodes: List[Episode], 
        gamma: float = 1.0
    ) -> Dict[Tuple[Any, ActionType], List[float]]:
        """
        Every-visit方式で状態行動ペアのリターンを計算
        
        Args:
            episodes: エピソードのリスト
            gamma: 割引率
            
        Returns:
            {(state, action): [returns]} の辞書
        """
        state_action_returns = {}
        
        for episode in episodes:
            for state, action, G in episode.get_state_action_returns(gamma):
                state_action_pair = (state, action)
                
                if state_action_pair not in state_action_returns:
                    state_action_returns[state_action_pair] = []
                
                state_action_returns[state_action_pair].append(G)
        
        return state_action_returns
    
    @staticmethod
    def calculate_q_values(
        state_action_returns: Dict[Tuple[Any, ActionType], List[float]]
    ) -> Dict[Tuple[Any, ActionType], float]:
        """
        リターンからQ値を計算（平均を取る）
        
        Args:
            state_action_returns: 状態行動ペアのリターン辞書
            
        Returns:
            Q値の辞書
        """
        q_values = {}
        
        for state_action_pair, returns in state_action_returns.items():
            q_values[state_action_pair] = np.mean(returns)
        
        return q_values
    
    @staticmethod
    def incremental_update(
        current_value: float, 
        new_return: float, 
        count: int
    ) -> float:
        """
        インクリメンタル平均更新
        
        Args:
            current_value: 現在の値
            new_return: 新しいリターン
            count: これまでの観測回数
            
        Returns:
            更新された値
        """
        return current_value + (new_return - current_value) / count