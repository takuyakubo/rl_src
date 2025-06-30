# 強化学習ライブラリ (rl-src)

強化学習のマルコフ決定過程（MDP）を学習するためのPythonライブラリです。
Zenn記事「強化学習コース」で使用されるサンプルコードを含んでいます。

## 特徴

このライブラリは以下の機能を提供します：

- **MDPCore中心アーキテクチャ**: データと実装の分離による一貫性保証
- **統一的な方策インターフェース**: Planning/Model-based/Model-freeの統一
- **GridWorld環境**: 学習用のシンプルな強化学習環境  
- **価値反復法**: 最適方策計算のプランニングアルゴリズム
- **実証コード**: マルコフ性と時不変性の検証

## 構成

```
rl_src/
├── examples/           # デモンストレーション
│   ├── gridworld.py          # GridWorld環境実装
│   ├── gridworld_factory.py  # MDPCore作成ファクトリー
│   ├── mdp_demonstration.py  # MDP基本概念のデモ
│   └── value_iteration_demo.py # 価値反復法プランニングデモ
├── demos/              # 高度なデモ
│   └── unified_agent_demo.py  # 統一エージェントアーキテクチャ
├── src/
│   ├── environments/   # 環境実装
│   │   ├── base.py           # 基底環境クラス
│   │   ├── mdp_core.py       # MDPCore（データクラス）
│   │   ├── mdp.py            # MDP環境（MDPCoreベース）
│   │   └── types.py          # 型定義
│   └── agents/         # エージェント実装
│       ├── base.py           # 基底エージェントクラス
│       ├── strategy_interfaces.py # 方策インターフェース
│       ├── strategies/       # 具体的戦略実装
│       │   ├── value_iteration.py  # 価値反復法
│       │   ├── q_learning.py       # Q学習
│       │   └── random_strategy.py  # ランダム戦略
│       └── types.py          # エージェント型定義
└── tests/              # テストコード
```

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/your_username/rl-src.git
cd rl-src

# 依存関係のインストール
uv install

# テストの実行
uv run pytest

# サンプル実行
uv run python examples/mdp_demonstration.py
```

## 使用例

### 基本的な使用方法（MDPCore中心）

```python
from examples.gridworld_factory import create_gridworld_mdp_core
from examples.gridworld import GridWorldEnvironment

# MDPCoreを作成
mdp_core = create_gridworld_mdp_core(
    size=3, 
    stochastic=False,
    goal=(2, 2),
    move_cost=-0.1,
    goal_reward=1.0
)

# 環境を作成
env = GridWorldEnvironment(mdp_core=mdp_core, goal=(2, 2))

# エピソードの実行
state = env.reset()
print(f"初期状態: {state}")

for step in range(10):
    if env.is_done():
        break
    
    # ランダムに行動を選択
    import random
    action = random.choice(env.get_action_space())
    
    # 行動を実行
    next_state, reward, done, info = env.step(action)
    print(f"ステップ{step+1}: {action} → {next_state}, 報酬={reward}")
    
    if done:
        print("🎉 ゴール到達！")
        break
```

### 価値反復法による最適方策計算

```python
from examples.gridworld_factory import create_gridworld_mdp_core
from src.agents.strategies.value_iteration import ValueIterationStrategy

# MDPCoreを作成
mdp_core = create_gridworld_mdp_core(size=3, stochastic=False)

# 価値反復法戦略を作成
strategy = ValueIterationStrategy(
    mdp_core=mdp_core,
    gamma=0.9,
    theta=1e-6
)

# 最適方策を計算
optimal_policy = strategy.plan()

# 特定の状態での行動分布を取得
observation = (0, 0)
available_actions = ['up', 'right', 'down', 'left']
action_probs = strategy.get_policy(observation, available_actions)
print(f"状態 {observation} での行動分布: {action_probs}")

# 最適行動を選択
action = strategy.get_action(observation, available_actions)
print(f"最適行動: {action}")
```

### 統一的な方策インターフェース

```python
from src.agents.strategies.q_learning import EpsilonGreedyStrategy
from src.agents.strategies.random_strategy import RandomStrategy

# 異なる戦略を同じインターフェースで使用
strategies = [
    ValueIterationStrategy(mdp_core=mdp_core, gamma=0.9),
    EpsilonGreedyStrategy(epsilon=0.1, learning_rate=0.1),
    RandomStrategy()
]

for strategy in strategies:
    # 統一インターフェース
    action_probs = strategy.get_policy(observation, available_actions)
    action = strategy.get_action(observation, available_actions)
    print(f"{type(strategy).__name__}: {action} (分布: {action_probs})")
```

## API リファレンス

### MDPCore中心アーキテクチャ

```python
# MDPCore - MDP定義の中心
@dataclass(frozen=True)
class MDPCore:
    states: List[StateType]                    # 状態空間
    actions: List[ActionType]                  # 行動空間
    transition_model: Callable                # 遷移確率関数
    reward_model: Callable                     # 報酬関数
    observation_to_state: Callable             # 観測→状態変換
```

### 方策インターフェース

```python
class PolicyStrategy(Protocol):
    def get_policy(self, observation, available_actions) -> Dict[ActionType, float]:
        """現在の観測での行動分布を取得"""
        
    def get_action(self, observation, available_actions) -> ActionType:
        """観測に基づいて行動を決定"""
        
    def update(self, experience: Experience) -> None:
        """経験から学習"""
```

### 戦略の種類

- **PlanningStrategy**: 事前計算による最適方策（価値反復法など）
- **ModelBasedStrategy**: 環境モデル学習による方策改善
- **ModelFreeStrategy**: 直接的な方策・価値学習（Q学習など）

## デモンストレーション

### 1. MDP基本概念の実証

```bash
uv run python examples/mdp_demonstration.py
```

- 決定的・確率的MDP環境の動作確認
- マルコフ性と時不変性の実証
- 異なる報酬設定での環境比較
- ランダム方策によるエピソード実行

### 2. 価値反復法プランニング

```bash
uv run python examples/value_iteration_demo.py
```

- 価値反復法による最適方策計算
- 異なる割引率での価値関数比較
- 確率的環境vs決定的環境の比較
- 学習された方策の実行デモ

### 3. 統一エージェントアーキテクチャ

```bash
uv run python demos/unified_agent_demo.py
```

- Planning/Model-free戦略の性能比較
- Dependency Injectionによる戦略注入
- エピソード管理と学習履歴分析

## 設計思想

### 1. MDPCore中心アーキテクチャ

MDPの定義（状態、行動、遷移、報酬）を`MDPCore`として分離し、環境と戦略が同じデータを共有：

```
MDPCore (データ) → Environment (シミュレーション)
             ↓
         Strategy (アルゴリズム)
```

### 2. 統一的な方策インターフェース

すべての戦略が同じインターフェースを実装し、相互に置き換え可能：

- `get_policy()`: 現在の観測での行動分布（確率>0のみ）
- `get_action()`: 方策に基づく行動選択
- `update()`: 経験からの学習

### 3. 型安全性とプロトコル

TypeScript風の型ヒントとプロトコルによる設計時契約の明示化。

## テスト

```bash
# 全テストの実行
uv run pytest

# 特定の性質のテスト
uv run pytest tests/test_gridworld.py -v

# カバレッジ付きテスト実行
uv run pytest --cov=src
```

## 拡張

### 新しい環境の追加

1. `MDPCore`を作成するファクトリー関数を実装
2. 必要に応じて`Environment`を継承した環境クラスを作成
3. テストケースを追加

### 新しい戦略の追加

1. `PlanningStrategy`、`ModelBasedStrategy`、`ModelFreeStrategy`のいずれかを継承
2. 必須メソッド（`get_policy`、`update_policy`など）を実装
3. `strategies`ディレクトリに追加

## 関連リンク

- [Zenn記事: 強化学習の基礎](https://zenn.dev/your_username/articles/d03abffb021dfa)
- [Zenn記事: マルコフ決定過程](https://zenn.dev/your_username/articles/362b23b4fb8da9)

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告は歓迎します。貢献方法：

1. フォークしてブランチを作成
2. 変更を実装してテストを追加
3. `uv run pytest`で全テストが通ることを確認
4. プルリクエストを作成

---

**注意**: このライブラリは教育目的で作成されており、実用的な強化学習アプリケーションには適さない場合があります。