"""
Transition modelのテスト

GridWorldEnvironmentのtransition_modelが正しく動作することを確認する。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gridworld import GridWorldEnvironment


def test_deterministic_transition_model():
    """決定的環境でのtransition_modelテスト"""
    print("=== 決定的環境でのtransition_modelテスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=False)
    
    # (0,0)からrightで(0,1)への遷移
    prob = env.transition_model((0, 0), "right", (0, 1))
    print(f"P((0,1) | (0,0), right) = {prob}")
    assert prob == 1.0, f"Expected 1.0, got {prob}"
    
    # (0,0)からrightで(0,2)への遷移（起こらない）
    prob = env.transition_model((0, 0), "right", (0, 2))
    print(f"P((0,2) | (0,0), right) = {prob}")
    assert prob == 0.0, f"Expected 0.0, got {prob}"
    
    # 境界での遷移：(0,0)からupで(0,0)への遷移（壁に当たる）
    prob = env.transition_model((0, 0), "up", (0, 0))
    print(f"P((0,0) | (0,0), up) = {prob}")
    assert prob == 1.0, f"Expected 1.0, got {prob}"
    
    print("決定的環境テスト: 成功")


def test_stochastic_transition_model():
    """確率的環境でのtransition_modelテスト"""
    print("\n=== 確率的環境でのtransition_modelテスト ===")
    
    env = GridWorldEnvironment(size=3, stochastic=True)
    
    # (1,1)からrightでの全遷移確率を表示
    print("(1,1)からrightでの遷移確率:")
    all_probs = {}
    for i in range(3):
        for j in range(3):
            prob = env.transition_model((1, 1), "right", (i, j))
            if prob > 0:
                all_probs[(i, j)] = prob
                print(f"  P(({i},{j}) | (1,1), right) = {prob:.6f}")
    
    # 遷移確率の合計が1になることを確認
    total_prob = sum(all_probs.values())
    print(f"  合計確率: {total_prob:.6f}")
    assert abs(total_prob - 1.0) < 1e-10, f"Expected 1.0, got {total_prob}"
    
    # 意図した遷移が最も高い確率を持つことを確認
    intended_prob = all_probs.get((1, 2), 0)
    other_probs = [prob for state, prob in all_probs.items() if state != (1, 2)]
    print(f"  意図した遷移の確率: {intended_prob:.6f}")
    print(f"  他の遷移の確率: {other_probs}")
    
    # 意図した遷移が他より高い確率を持つことを確認
    if other_probs:
        assert intended_prob > max(other_probs), "意図した遷移の確率が最高でない"
    
    print("確率的環境テスト: 成功")


def test_transition_probabilities_sum_to_one():
    """遷移確率の合計が1になることを確認"""
    print("\n=== 遷移確率の合計テスト ===")
    
    for stochastic in [False, True]:
        env = GridWorldEnvironment(size=3, stochastic=stochastic)
        env_type = "確率的" if stochastic else "決定的"
        
        # (1,1)からrightの全遷移確率の合計
        total_prob = 0.0
        for i in range(3):
            for j in range(3):
                prob = env.transition_model((1, 1), "right", (i, j))
                total_prob += prob
        
        print(f"{env_type}環境: P(s' | (1,1), right) の合計 = {total_prob}")
        assert abs(total_prob - 1.0) < 1e-10, f"Expected 1.0, got {total_prob}"
    
    print("遷移確率合計テスト: 成功")


if __name__ == "__main__":
    test_deterministic_transition_model()
    test_stochastic_transition_model()
    test_transition_probabilities_sum_to_one()
    
    print("\n" + "=" * 50)
    print("全てのtransition_modelテストが成功しました！")
    print("=" * 50)