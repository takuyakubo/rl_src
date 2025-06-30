"""
エピソード管理

複数エピソードにわたる履歴の管理と分析を行う。
"""

from typing import List, Dict, Any, Optional
from .types import Experience, Episode, AgentHistory


class EpisodeManager:
    """
    エピソード管理クラス
    
    複数エピソードにわたる履歴の管理、統計の計算、分析機能を提供する。
    """
    
    def __init__(self):
        """エピソード管理を初期化"""
        self.history = AgentHistory()
    
    def add_episode_experiences(self, experiences: List[Experience]) -> Episode:
        """
        エピソードの経験リストを追加
        
        Args:
            experiences: エピソードの経験リスト
            
        Returns:
            作成されたエピソード
        """
        if not experiences:
            return Episode(experiences=[], total_reward=0.0, episode_length=0)
        
        total_reward = sum(exp.reward for exp in experiences)
        episode = Episode(
            experiences=experiences,
            total_reward=total_reward,
            episode_length=len(experiences)
        )
        
        self.history.episodes.append(episode)
        return episode
    
    def get_total_episodes(self) -> int:
        """総エピソード数を取得"""
        return len(self.history.episodes)
    
    def get_average_reward(self) -> float:
        """平均エピソード報酬を取得"""
        if not self.history.episodes:
            return 0.0
        
        total_reward = sum(episode.total_reward for episode in self.history.episodes)
        return total_reward / len(self.history.episodes)
    
    def get_average_length(self) -> float:
        """平均エピソード長を取得"""
        if not self.history.episodes:
            return 0.0
        
        total_length = sum(episode.episode_length for episode in self.history.episodes)
        return total_length / len(self.history.episodes)
    
    def get_last_n_episodes(self, n: int) -> List[Episode]:
        """最新のN個のエピソードを取得"""
        return self.history.episodes[-n:] if n > 0 else []
    
    def get_reward_history(self) -> List[float]:
        """各エピソードの報酬履歴を取得"""
        return [episode.total_reward for episode in self.history.episodes]
    
    def get_length_history(self) -> List[int]:
        """各エピソードの長さ履歴を取得"""
        return [episode.episode_length for episode in self.history.episodes]
    
    def get_success_rate(self, success_threshold: float = 0.0) -> float:
        """
        成功率を計算
        
        Args:
            success_threshold: 成功とみなす報酬の閾値
            
        Returns:
            成功率（0.0〜1.0）
        """
        if not self.history.episodes:
            return 0.0
        
        successful_episodes = sum(
            1 for episode in self.history.episodes 
            if episode.total_reward > success_threshold
        )
        return successful_episodes / len(self.history.episodes)
    
    def get_recent_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        最近のN個のエピソードの性能統計を取得
        
        Args:
            n_episodes: 統計を計算するエピソード数
            
        Returns:
            性能統計の辞書
        """
        recent_episodes = self.get_last_n_episodes(n_episodes)
        
        if not recent_episodes:
            return {
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "success_rate": 0.0,
                "episode_count": 0
            }
        
        avg_reward = sum(ep.total_reward for ep in recent_episodes) / len(recent_episodes)
        avg_length = sum(ep.episode_length for ep in recent_episodes) / len(recent_episodes)
        success_rate = sum(1 for ep in recent_episodes if ep.total_reward > 0) / len(recent_episodes)
        
        return {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "success_rate": success_rate,
            "episode_count": len(recent_episodes)
        }
    
    def clear_history(self) -> None:
        """履歴をクリア"""
        self.history = AgentHistory()
    
    def print_summary(self, recent_n: int = 10) -> None:
        """
        履歴の概要を表示
        
        Args:
            recent_n: 最近の統計を計算するエピソード数
        """
        total_episodes = self.get_total_episodes()
        print(f"=== エピソード履歴概要 ===")
        print(f"総エピソード数: {total_episodes}")
        
        if total_episodes > 0:
            print(f"全期間統計:")
            print(f"  平均報酬: {self.get_average_reward():.3f}")
            print(f"  平均エピソード長: {self.get_average_length():.1f}")
            print(f"  成功率: {self.get_success_rate():.1%}")
            
            if total_episodes >= recent_n:
                recent_stats = self.get_recent_performance(recent_n)
                print(f"最近{recent_n}エピソード:")
                print(f"  平均報酬: {recent_stats['avg_reward']:.3f}")
                print(f"  平均エピソード長: {recent_stats['avg_length']:.1f}")
                print(f"  成功率: {recent_stats['success_rate']:.1%}")