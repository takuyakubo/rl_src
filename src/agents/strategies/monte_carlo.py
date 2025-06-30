"""
モンテカルロ法戦略の実装

First-visit Monte Carlo方法とMonte Carlo Controlを提供する。
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..strategy_interfaces import ModelFreeStrategy
from ..types import Experience, ObservationType, ActionType
from ..episode_generation import EpisodeGenerator, MonteCarloUtils
from ...environments.base import Environment


class FirstVisitMonteCarloStrategy(ModelFreeStrategy):
    """
    First-visit Monte Carlo方法による価値関数学習
    
    エピソードを生成して、各状態行動ペアの初回訪問時のリターンから
    Q値を学習する。
    """
    
    def __init__(
        self,
        environment: Environment,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        First-visit Monte Carlo戦略を初期化
        
        Args:
            environment: 学習環境
            epsilon: ε-greedy方策の探索率
            gamma: 割引率
            random_seed: 乱数シード
        """
        super().__init__(gamma=gamma, **kwargs)
        self.environment = environment
        self.epsilon = epsilon
        
        # Q値とカウンタ
        self.Q: Dict[Tuple[Any, ActionType], float] = defaultdict(float)
        self.returns: Dict[Tuple[Any, ActionType], List[float]] = defaultdict(list)
        self.visit_counts: Dict[Tuple[Any, ActionType], int] = defaultdict(int)
        
        # エピソード生成器
        self.episode_generator = EpisodeGenerator(
            environment=environment,
            random_seed=random_seed
        )
        
        # 学習統計
        self.episodes_generated = 0
        self.last_episode_reward = 0.0
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """
        ε-greedy方策による行動分布を取得
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
            
        Returns:
            行動分布（{行動: 確率} の辞書）
        """
        state = self._observation_to_state(observation)
        
        # Q値から最適行動を決定
        best_action = None
        best_q_value = float('-inf')
        
        for action in available_actions:
            q_value = self.Q.get((state, action), 0.0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        if best_action is None:
            # Q値がない場合は均等分布
            uniform_prob = 1.0 / len(available_actions)
            return {action: uniform_prob for action in available_actions}
        
        # ε-greedy方策の確率分布
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
    
    def _observation_to_state(self, observation: ObservationType) -> Any:
        """観測を状態に変換"""
        return observation
    
    def update_policy(self, experience: Experience) -> None:
        """
        経験から方策を更新（Monte Carlo法では使用しない）
        
        Monte Carlo法はエピソード単位で学習するため、
        このメソッドは使用しない。代わりにlearn_from_episodes()を使用。
        """
        pass
    
    def learn_from_episodes(self, num_episodes: int) -> Dict[str, Any]:
        """
        指定された数のエピソードから学習
        
        Args:
            num_episodes: 生成するエピソード数
            
        Returns:
            学習統計情報
        """
        # 現在の方策関数を作成
        def policy_func(state, available_actions):
            return self.get_action(state, available_actions)
        
        # エピソードを生成
        episodes = self.episode_generator.generate_episodes(
            policy_func=policy_func,
            num_episodes=num_episodes
        )
        
        # First-visit方式でリターンを計算
        state_action_returns = MonteCarloUtils.first_visit_returns(
            episodes, self.gamma
        )
        
        # Q値を更新
        updates_made = 0
        for (state, action), returns in state_action_returns.items():
            for return_value in returns:
                self.returns[(state, action)].append(return_value)
                self.visit_counts[(state, action)] += 1
                
                # インクリメンタル平均でQ値を更新
                count = self.visit_counts[(state, action)]
                self.Q[(state, action)] = MonteCarloUtils.incremental_update(
                    self.Q[(state, action)], return_value, count
                )
                updates_made += 1
        
        # 学習統計を更新
        self.episodes_generated += num_episodes
        self.learning_updates += updates_made
        
        if episodes:
            self.last_episode_reward = episodes[-1].total_reward
        
        # 統計情報を返す
        avg_episode_length = np.mean([ep.length for ep in episodes]) if episodes else 0
        avg_episode_reward = np.mean([ep.total_reward for ep in episodes]) if episodes else 0
        
        return {
            'episodes_generated': num_episodes,
            'updates_made': updates_made,
            'avg_episode_length': avg_episode_length,
            'avg_episode_reward': avg_episode_reward,
            'total_episodes': self.episodes_generated,
            'unique_state_actions': len(self.Q),
            'total_visits': sum(self.visit_counts.values())
        }
    
    def get_q_value(self, state: Any, action: ActionType) -> float:
        """特定の状態行動ペアのQ値を取得"""
        return self.Q.get((state, action), 0.0)
    
    def get_all_q_values(self) -> Dict[Tuple[Any, ActionType], float]:
        """全てのQ値を取得"""
        return dict(self.Q)
    
    def get_value_function(self) -> Dict[Any, float]:
        """状態価値関数を取得（最大Q値）"""
        value_function = {}
        
        # 状態ごとに最大Q値を計算
        states = set(state for state, action in self.Q.keys())
        for state in states:
            state_q_values = [
                q_value for (s, a), q_value in self.Q.items() if s == state
            ]
            if state_q_values:
                value_function[state] = max(state_q_values)
            else:
                value_function[state] = 0.0
        
        return value_function


class MonteCarloControlStrategy(FirstVisitMonteCarloStrategy):
    """
    Monte Carlo Control（方策改善付きMonte Carlo）
    
    ε-greedy方策を使って探索しながら、Q値の学習と方策の改善を
    同時に行う。
    """
    
    def __init__(
        self,
        environment: Environment,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        gamma: float = 1.0,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Monte Carlo Control戦略を初期化
        
        Args:
            environment: 学習環境
            epsilon: 初期ε-greedy探索率
            epsilon_decay: εの減衰率
            min_epsilon: εの最小値
            gamma: 割引率
            random_seed: 乱数シード
        """
        super().__init__(
            environment=environment,
            epsilon=epsilon,
            gamma=gamma,
            random_seed=random_seed,
            **kwargs
        )
        
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
    
    def learn_from_episodes(self, num_episodes: int) -> Dict[str, Any]:
        """
        エピソードから学習し、方策を改善
        
        Args:
            num_episodes: 生成するエピソード数
            
        Returns:
            学習統計情報
        """
        # 基底クラスの学習を実行
        stats = super().learn_from_episodes(num_episodes)
        
        # ε値を減衰
        old_epsilon = self.epsilon
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )
        
        # 統計情報を追加
        stats.update({
            'epsilon_before': old_epsilon,
            'epsilon_after': self.epsilon,
            'epsilon_decayed': old_epsilon != self.epsilon
        })
        
        return stats
    
    def reset_exploration(self):
        """探索率をリセット"""
        self.epsilon = self.initial_epsilon


class MonteCarloEvaluationStrategy(ModelFreeStrategy):
    """
    Monte Carlo方策評価（固定方策の価値関数学習）
    
    与えられた固定方策に従ってエピソードを生成し、
    その方策の価値関数を学習する。
    """
    
    def __init__(
        self,
        environment: Environment,
        target_policy_func: callable,
        gamma: float = 1.0,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Monte Carlo評価戦略を初期化
        
        Args:
            environment: 学習環境
            target_policy_func: 評価対象の方策関数
            gamma: 割引率
            random_seed: 乱数シード
        """
        super().__init__(gamma=gamma, **kwargs)
        self.environment = environment
        self.target_policy_func = target_policy_func
        
        # Q値とカウンタ
        self.Q: Dict[Tuple[Any, ActionType], float] = defaultdict(float)
        self.returns: Dict[Tuple[Any, ActionType], List[float]] = defaultdict(list)
        self.visit_counts: Dict[Tuple[Any, ActionType], int] = defaultdict(int)
        
        # エピソード生成器
        self.episode_generator = EpisodeGenerator(
            environment=environment,
            random_seed=random_seed
        )
        
        # 学習統計
        self.episodes_generated = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def get_policy(self, observation: ObservationType, available_actions: List[ActionType]) -> Dict[ActionType, float]:
        """
        評価対象の方策による行動分布を取得
        
        Args:
            observation: 現在の観測
            available_actions: 利用可能な行動のリスト
            
        Returns:
            行動分布（評価対象方策に従う）
        """
        action = self.target_policy_func(observation, available_actions)
        return {action: 1.0}
    
    def _observation_to_state(self, observation: ObservationType) -> Any:
        """観測を状態に変換"""
        return observation
    
    def update_policy(self, experience: Experience) -> None:
        """方策は固定なので更新しない"""
        pass
    
    def evaluate_policy(self, num_episodes: int) -> Dict[str, Any]:
        """
        固定方策を評価
        
        Args:
            num_episodes: 生成するエピソード数
            
        Returns:
            評価統計情報
        """
        # エピソードを生成
        episodes = self.episode_generator.generate_episodes(
            policy_func=self.target_policy_func,
            num_episodes=num_episodes
        )
        
        # First-visit方式でリターンを計算
        state_action_returns = MonteCarloUtils.first_visit_returns(
            episodes, self.gamma
        )
        
        # Q値を更新
        updates_made = 0
        for (state, action), returns in state_action_returns.items():
            for return_value in returns:
                self.returns[(state, action)].append(return_value)
                self.visit_counts[(state, action)] += 1
                
                # インクリメンタル平均でQ値を更新
                count = self.visit_counts[(state, action)]
                self.Q[(state, action)] = MonteCarloUtils.incremental_update(
                    self.Q[(state, action)], return_value, count
                )
                updates_made += 1
        
        # 学習統計を更新
        self.episodes_generated += num_episodes
        self.learning_updates += updates_made
        
        # 統計情報を返す
        avg_episode_length = np.mean([ep.length for ep in episodes]) if episodes else 0
        avg_episode_reward = np.mean([ep.total_reward for ep in episodes]) if episodes else 0
        
        return {
            'episodes_evaluated': num_episodes,
            'updates_made': updates_made,
            'avg_episode_length': avg_episode_length,
            'avg_episode_reward': avg_episode_reward,
            'total_episodes': self.episodes_generated,
            'unique_state_actions': len(self.Q),
            'total_visits': sum(self.visit_counts.values())
        }
    
    def get_value_function(self) -> Dict[Any, float]:
        """状態価値関数を取得"""
        value_function = {}
        
        # 状態ごとに価値を計算（方策に従った期待値）
        states = set(state for state, action in self.Q.keys())
        for state in states:
            # この状態での方策による期待価値を計算
            available_actions = self.environment.get_available_actions()
            
            if available_actions:
                action_probs = self.get_policy(state, available_actions)
                expected_value = 0.0
                
                for action, prob in action_probs.items():
                    q_value = self.Q.get((state, action), 0.0)
                    expected_value += prob * q_value
                
                value_function[state] = expected_value
            else:
                value_function[state] = 0.0
        
        return value_function