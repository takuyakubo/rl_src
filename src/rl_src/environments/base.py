"""
環境の抽象基底クラス

全ての強化学習環境が実装すべき基本的なインターフェースを定義する。
特定の理論的枠組み（MDPなど）には依存しない汎用的な設計。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Environment(ABC):
    """
    強化学習環境の抽象基底クラス
    
    エージェントと環境の相互作用を定義する最小限のインターフェース。
    どの理論的枠組み（MDP、POMDP、マルチエージェントなど）でも利用可能。
    """
    
    def __init__(self):
        """環境を初期化する"""
        self._history: List[Dict[str, Any]] = []
        self._episode_done: bool = False
    
    @abstractmethod
    def reset(self, **kwargs) -> Any:
        """
        環境をリセットして初期観測を返す
        
        Args:
            **kwargs: 環境固有のリセット引数
            
        Returns:
            初期観測
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """
        行動を実行して結果を返す
        
        Args:
            action: 実行する行動
            
        Returns:
            observation: 次の観測
            reward: 受け取った報酬
            done: エピソード終了フラグ
            info: 追加情報
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> List[Any]:
        """
        利用可能な行動の集合を返す
        
        Returns:
            行動集合のリスト
        """
        pass
    
    def get_observation_space(self) -> Optional[List[Any]]:
        """
        観測空間を返す（オプション）
        
        Returns:
            観測空間のリスト、定義されていない場合はNone
        """
        return None
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        エピソードの履歴を返す
        
        Returns:
            履歴のリスト。各要素は {"observation", "action", "reward", "done", "info"} を含む
        """
        return self._history.copy()
    
    def is_done(self) -> bool:
        """エピソードが終了しているかを返す"""
        return self._episode_done
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        環境の現在の状態を可視化する
        
        Args:
            mode: 描画モード
            
        Returns:
            mode に応じた描画結果
        """
        pass
    
    def close(self) -> None:
        """環境のリソースを解放する"""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        環境の乱数シードを設定する
        
        Args:
            seed: 乱数シード
        """
        pass
    
    # === 継承クラス用のヘルパーメソッド ===
    
    def _reset_history(self) -> None:
        """履歴をリセットする"""
        self._history = []
        self._episode_done = False
    
    def _add_to_history(self, observation: Any, action: Any, reward: float, 
                       done: bool, info: Dict[str, Any]) -> None:
        """履歴にエントリを追加する"""
        self._history.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        })
        self._episode_done = done


class DiscreteEnvironment(Environment):
    """
    離散的な状態・行動空間を持つ環境の基底クラス
    
    状態空間が明確に定義できる環境で使用する。
    """
    
    @abstractmethod
    def get_state_space(self) -> List[Any]:
        """
        状態空間を返す
        
        Returns:
            状態集合のリスト
        """
        pass
    
    def get_current_state(self) -> Optional[Any]:
        """
        現在の内部状態を返す（デバッグ用）
        
        Returns:
            現在の状態、未定義の場合はNone
        """
        return None


class MDPEnvironment(DiscreteEnvironment):
    """
    マルコフ決定過程として定式化された環境の基底クラス
    
    MDP特有の概念（状態遷移確率、報酬関数など）を扱う環境で使用する。
    """
    
    @abstractmethod
    def get_transition_model(self) -> Dict[Any, Dict[Any, Dict[Any, float]]]:
        """
        状態遷移モデル P(s'|s,a) を返す
        
        Returns:
            遷移確率の辞書 {state: {action: {next_state: probability}}}
        """
        pass
    
    @abstractmethod
    def get_reward_model(self) -> Dict[Any, Dict[Any, float]]:
        """
        報酬モデル R(r|s,a) を返す
        
        Returns:
            報酬の辞書 {state: {action: reward}}
        """
        pass