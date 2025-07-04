"""
環境パッケージ

強化学習における環境の実装を提供する。
全ての環境はEnvironment抽象基底クラスを継承し、
統一されたインターフェースを持つ。
"""

from .base import Environment

__all__ = ["Environment"]