# 強化学習ライブラリ (rl-src)

強化学習のマルコフ決定過程（MDP）を学習するためのPythonライブラリです。
Zenn記事「強化学習コース」で使用されるサンプルコードを含んでいます。

## 特徴

このライブラリは以下の機能を提供します：

- **環境インターフェース**: MDPの理論に基づく統一的な環境API
- **グリッドワールド**: 学習用のシンプルな強化学習環境  
- **可視化機能**: 環境の状態とエージェントの軌跡を可視化
- **検証ツール**: マルコフ性と時不変性の実証コード

## 構成

### 1. 環境インターフェース
- 抽象基底クラス `Environment` による統一インターフェース
- マルコフ性と時不変性を保証する設計
- 階層的な環境クラス構造

### 2. MDP実装
- マルコフ性の実装（現在の状態のみに依存）
- 時不変性の実装（時刻に依存しない遷移）
- 確率的・決定的両環境への対応

### 3. 検証機能
- マルコフ性の実証
- MDPの性質の可視化
- エピソード分析機能

## インストール

このライブラリは以下の手順でセットアップできます：

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

### 基本的な使用方法

```python
from src.rl_src.environments import GridWorldEnvironment

# 環境の作成
env = GridWorldEnvironment(size=3, stochastic=False)

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

# エピソード分析
analysis = env.analyze_episode()
print(f"総ステップ数: {analysis['total_steps']}")
print(f"総報酬: {analysis['total_reward']:.1f}")
```

### MDPの性質の検証

```python
# マルコフ性の実証
env.demonstrate_markov_property()

# 時不変性の実証  
env.demonstrate_time_invariance()

# 遷移確率の確認
transition_model = env.get_transition_model()
state = (1, 1)
action = "right"
print(f"P(s'|{state}, {action}) = {transition_model[state][action]}")
```

## API リファレンス

### 環境クラスの階層

```
Environment (抽象基底クラス)
├── reset() - 環境のリセット
├── step() - 行動の実行
├── render() - 状態の可視化
└── get_action_space() - 利用可能な行動

DiscreteEnvironment (離散環境)
├── get_state_space() - 状態空間
├── get_transition_model() - 遷移確率 P(s'|s,a)
└── get_reward_model() - 報酬関数 R(r|s,a)

GridWorldEnvironment (グリッドワールド)
├── マルコフ性と時不変性を満たすMDP
├── 確率的・決定的遷移の選択可能
└── 可視化機能
```

### MDPの構成要素 

```python
# MDP = ⟨S, A, P, R⟩
S = env.get_state_space()        # 状態集合
A = env.get_action_space()       # 行動集合  
P = env.get_transition_model()   # 状態遷移確率 P(s'|s,a)
R = env.get_reward_model()       # 報酬関数 R(r|s,a)
```

## 実装詳細

### GridWorldEnvironment

N×Nのグリッド上でエージェントが移動し、ゴールを目指す環境です。

**主要パラメータ**:
- `size`: グリッドのサイズ（デフォルト: 3）
- `goal`: ゴールの位置（デフォルト: (2,2)）
- `stochastic`: 確率的遷移の使用（デフォルト: False）
- `move_cost`: 移動コスト（デフォルト: -0.1）
- `goal_reward`: ゴール報酬（デフォルト: 1.0）

**主要機能**:
- 境界での反射処理
- 確率的遷移（意図した方向に80%、他方向に各5%）
- エピソード履歴の管理
- 可視化機能（テキスト・RGB配列）

## テスト

```bash
# 全テストの実行
uv run pytest

# 特定の性質のテスト
uv run pytest tests/test_gridworld.py::TestMDPProperties -v

# カバレッジ付きテスト実行
uv run pytest --cov=src
```

### テスト内容

- **基本機能**: 環境のリセットと行動実行
- **MDP性質**: マルコフ性と時不変性の確認
- **確率分布**: 遷移確率の正規性と合計が1
- **エラーハンドリング**: 無効な行動と状態

## サンプル実行

```bash
uv run python examples/mdp_demonstration.py
```

このコマンドで以下が実行されます：

1. **決定的環境**: 基本的なMDPの動作確認
2. **確率的環境**: 確率的遷移の動作確認
3. **マルコフ性実証**: 履歴に依存しない遷移確率
4. **時不変性実証**: 時刻に依存しない規則
5. **比較実験**: 異なる環境設定での結果比較
6. **可視化デモ**: 環境のエピソード実行例

## 拡張

### 新しい環境の追加

1. `Environment`または`DiscreteEnvironment`を継承
2. 必須メソッド（`reset`, `step`, `get_action_space`）を実装
3. `src/rl_src/environments/__init__.py`でエクスポート
4. テストケースを追加

### 新しいアルゴリズムの追加

1. `src/rl_src/agents/`ディレクトリに追加
2. 抽象基底クラス`Agent`を作成
3. 価値反復法やQ学習などのMDP解法を実装

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