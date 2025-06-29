"""
MDP（マルコフ決定過程）の実証

Zenn記事で説明したMDPの概念を実際のコードで体験する。
GridWorldEnvironmentを使ってマルコフ性と時不変性を確認する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.rl_src.environments import GridWorldEnvironment


def main():
    """MDP実証のメイン関数"""
    print("=" * 60)
    print("MDP（マルコフ決定過程）の実証プログラム")
    print("=" * 60)
    
    # 1. 決定的環境でのMDP実証
    demonstrate_deterministic_mdp()
    
    print("\\n" + "=" * 60)
    
    # 2. 確率的環境でのMDP実証
    demonstrate_stochastic_mdp()
    
    print("\\n" + "=" * 60)
    
    # 3. 環境の比較実験
    compare_environments()
    
    print("\\n" + "=" * 60)
    
    # 4. エピソード実行例
    run_sample_episodes()


def demonstrate_deterministic_mdp():
    """決定的MDPの実証"""
    print("\\n=== 決定的MDP環境 ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False, random_seed=42)
    
    # 基本情報を表示
    print(f"状態空間のサイズ: {len(env.get_state_space())}")
    print(f"行動空間: {env.get_action_space()}")
    print(f"ゴール位置: {env.goal}")
    
    # マルコフ性の実証
    env.demonstrate_markov_property()
    
    # 時不変性の実証
    env.demonstrate_time_invariance()
    
    # 実際のエピソード実行
    print("\\n--- エピソード実行例 ---")
    env.reset()
    print(f"初期状態: {env.get_current_state()}")
    
    actions = ["right", "right", "down", "down"]
    for i, action in enumerate(actions):
        if not env.is_done():
            next_state, reward, done, info = env.step(action)
            print(f"ステップ{i+1}: {action} → {next_state}, 報酬={reward:.1f}, 終了={done}")
        
        if env.is_done():
            break
    
    # エピソード分析
    analysis = env.analyze_episode()
    print(f"\\nエピソード結果:")
    print(f"  総ステップ数: {analysis['total_steps']}")
    print(f"  総報酬: {analysis['total_reward']:.1f}")
    print(f"  ゴール到達: {analysis['reached_goal']}")


def demonstrate_stochastic_mdp():
    """確率的MDPの実証"""
    print("\\n=== 確率的MDP環境 ===")
    
    env = GridWorldEnvironment(size=3, stochastic=True, random_seed=42)
    
    # マルコフ性の実証（確率的環境でも同じ）
    env.demonstrate_markov_property()
    
    # 確率分布の詳細確認
    print("\\n--- 確率的遷移の詳細 ---")
    test_state = (1, 1)
    test_action = "right"
    
    transition_model = env.get_transition_model()
    prob_dist = transition_model[test_state][test_action]
    
    print(f"状態 {test_state} で行動 '{test_action}' を取った場合:")
    for next_state, probability in prob_dist.items():
        if probability > 0:
            print(f"  → {next_state}: {probability:.1f}")
    
    print(f"確率の総和: {sum(prob_dist.values()):.1f}")
    
    # 複数回実行して確率的性質を確認
    print("\\n--- 確率的実行の例 ---")
    results = []
    for episode in range(5):
        env.reset(start_position=test_state)
        next_state, reward, done, info = env.step(test_action)
        results.append(next_state)
        print(f"実行{episode+1}: {test_state} --{test_action}--> {next_state}")
    
    # 結果の分布を確認
    unique_results = set(results)
    print(f"\\n観測された次状態: {unique_results}")
    if len(unique_results) > 1:
        print("→ 確率的環境では異なる結果が観測される")
    else:
        print("→ この実行では同じ結果（確率的でも起こりうる）")


def compare_environments():
    """環境の比較実験"""
    print("\\n=== 環境の比較実験 ===")
    
    # 異なる設定の環境を作成
    environments = {
        "決定的環境": GridWorldEnvironment(size=3, stochastic=False, random_seed=42),
        "確率的環境": GridWorldEnvironment(size=3, stochastic=True, random_seed=42),
        "大きなコスト": GridWorldEnvironment(size=3, stochastic=False, move_cost=-0.5, random_seed=42),
        "大きな報酬": GridWorldEnvironment(size=3, stochastic=False, goal_reward=10.0, random_seed=42)
    }
    
    test_trajectory = ["right", "down", "right", "down"]
    
    print("同じ軌跡での各環境の比較:")
    print("軌跡:", " → ".join(test_trajectory))
    print()
    
    for env_name, env in environments.items():
        env.reset()
        total_reward = 0
        
        print(f"{env_name}:")
        for action in test_trajectory:
            if not env.is_done():
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                print(f"  {action}: 報酬={reward:.1f}")
                
                if done:
                    print(f"  → ゴール到達！")
                    break
        
        print(f"  総報酬: {total_reward:.1f}")
        print()


def run_sample_episodes():
    """サンプルエピソードの実行"""
    print("\\n=== サンプルエピソードの実行 ===")
    
    env = GridWorldEnvironment(size=4, stochastic=False, random_seed=42)
    
    # より大きなグリッドでのエピソード実行
    print("4x4グリッドでのランダム方策エピソード:")
    
    np.random.seed(42)
    
    for episode in range(3):
        print(f"\\n--- エピソード {episode + 1} ---")
        
        env.reset()
        step_count = 0
        max_steps = 20
        
        print(f"開始位置: {env.get_current_state()}")
        
        while not env.is_done() and step_count < max_steps:
            # ランダムに行動を選択
            action = np.random.choice(env.get_action_space())
            next_state, reward, done, info = env.step(action)
            
            step_count += 1
            print(f"ステップ{step_count}: {action} → {next_state}, 報酬={reward:.1f}")
            
            if done:
                print("🎉 ゴール到達！")
                break
        
        if not env.is_done():
            print("⏰ 最大ステップ数に到達")
        
        # エピソード分析
        analysis = env.analyze_episode()
        print(f"結果: {analysis['total_steps']}ステップ, 総報酬={analysis['total_reward']:.1f}")


def visualize_environment():
    """環境の可視化（オプション）"""
    print("\\n=== 環境の可視化 ===")
    
    env = GridWorldEnvironment(size=4, stochastic=False)
    env.reset()
    
    # いくつかの行動を実行
    actions = ["right", "right", "down", "left", "down", "right"]
    
    for action in actions:
        if not env.is_done():
            env.step(action)
    
    try:
        # RGB配列として描画
        rgb_array = env.render(mode="rgb_array")
        print(f"環境を可視化しました（配列サイズ: {rgb_array.shape}）")
        
        # ファイルに保存
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_array)
        plt.title("GridWorld Environment")
        plt.axis('off')
        plt.savefig("gridworld_visualization.png", dpi=150, bbox_inches='tight')
        print("可視化結果を 'gridworld_visualization.png' に保存しました")
        
    except Exception as e:
        print(f"可視化でエラーが発生しました: {e}")
        print("matplotlib が正しく設定されていない可能性があります")


if __name__ == "__main__":
    main()
    
    # 可視化も実行（オプション）
    try:
        visualize_environment()
    except ImportError:
        print("\\nmatplotlib が利用できないため、可視化をスキップしました")
    except Exception as e:
        print(f"\\n可視化でエラーが発生しました: {e}")
    
    print("\\n" + "=" * 60)
    print("MDP実証プログラム完了")
    print("詳細は test_gridworld.py でテストできます")
    print("=" * 60)