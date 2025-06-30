"""
エージェントパッケージ

強化学習アルゴリズムの実装を提供する。
観測ベースで動作し、方策決定戦略をDIで注入する統一的な設計。
"""

# 統一的なエージェント設計
from .base import Agent
from .episode_manager import EpisodeManager
from .types import Experience, Episode, AgentHistory, ObservationType, ActionType
from .strategy_interfaces import (
    PolicyStrategy, PlanningStrategy, ModelBasedStrategy, ModelFreeStrategy
)
from .strategies import ValueIterationStrategy, EpsilonGreedyStrategy, RandomStrategy


__all__ = [
    # 統一的なエージェント設計
    "Agent",
    "PolicyStrategy",
    "EpisodeManager",
    
    # 戦略基底クラス
    "PlanningStrategy",
    "ModelBasedStrategy",
    "ModelFreeStrategy",
    
    # 具体的な戦略
    "ValueIterationStrategy",
    "EpsilonGreedyStrategy",
    "RandomStrategy",
    
    # 型定義
    "Experience",
    "Episode", 
    "AgentHistory",
    "ObservationType",
    "ActionType"
]