"""
エージェントで使用する型定義

エージェントの経験と履歴管理のための型、および共通の型変数を定義する。
"""

from typing import Any, Dict, Optional, TypeVar
from pydantic import BaseModel, Field

# 共通型変数の定義
ObservationType = TypeVar('ObservationType')
ActionType = TypeVar('ActionType')


class Experience(BaseModel):
    """
    エージェントの単一経験
    
    エージェントが環境との相互作用で得る一つの経験を表す。
    observation -> action -> reward -> next_observation の流れを記録。
    """
    observation: Any = Field(description="現在の観測")
    action: Any = Field(description="実行した行動")
    reward: float = Field(description="受け取った報酬")
    next_observation: Any = Field(description="次の観測")
    done: bool = Field(description="エピソード終了フラグ")
    info: Dict[str, Any] = Field(default_factory=dict, description="追加情報")

    class Config:
        arbitrary_types_allowed = True


class Episode(BaseModel):
    """
    エピソード（一連の経験）
    
    開始から終了までの経験の連続を表す。
    """
    experiences: list[Experience] = Field(default_factory=list, description="経験のリスト")
    total_reward: float = Field(default=0.0, description="エピソードの総報酬")
    episode_length: int = Field(default=0, description="エピソードの長さ")
    
    def add_experience(self, experience: Experience) -> None:
        """経験を追加し、統計を更新"""
        self.experiences.append(experience)
        self.total_reward += experience.reward
        self.episode_length += 1
    
    def is_complete(self) -> bool:
        """エピソードが完了しているかチェック"""
        return len(self.experiences) > 0 and self.experiences[-1].done

    class Config:
        arbitrary_types_allowed = True


class AgentHistory(BaseModel):
    """
    エージェントの全履歴
    
    エージェントが経験した全エピソードと統計情報を管理。
    """
    episodes: list[Episode] = Field(default_factory=list, description="エピソードのリスト")
    current_episode: Optional[Episode] = Field(default=None, description="現在のエピソード")
    
    def start_new_episode(self) -> None:
        """新しいエピソードを開始"""
        if self.current_episode is not None and not self.current_episode.is_complete():
            # 前のエピソードが未完了の場合は保存
            self.episodes.append(self.current_episode)
        
        self.current_episode = Episode()
    
    def add_experience(self, experience: Experience) -> None:
        """現在のエピソードに経験を追加"""
        if self.current_episode is None:
            self.start_new_episode()
        
        self.current_episode.add_experience(experience)
        
        # エピソードが完了したら履歴に追加
        if experience.done:
            self.episodes.append(self.current_episode)
            self.current_episode = None
    
    def get_total_episodes(self) -> int:
        """総エピソード数を取得"""
        return len(self.episodes)
    
    def get_average_reward(self) -> float:
        """平均総報酬を取得"""
        if not self.episodes:
            return 0.0
        return sum(ep.total_reward for ep in self.episodes) / len(self.episodes)
    
    def get_average_length(self) -> float:
        """平均エピソード長を取得"""
        if not self.episodes:
            return 0.0
        return sum(ep.episode_length for ep in self.episodes) / len(self.episodes)
    
    def get_last_n_episodes(self, n: int) -> list[Episode]:
        """最新のn個のエピソードを取得"""
        return self.episodes[-n:] if n <= len(self.episodes) else self.episodes
    
    def clear_history(self) -> None:
        """履歴をクリア"""
        self.episodes = []
        self.current_episode = None

    class Config:
        arbitrary_types_allowed = True