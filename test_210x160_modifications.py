"""
VERIFICATION SCRIPT - 210x160 Grayscale Input
==============================================

Run this script to verify the modifications work correctly.
Tests input shapes and network forward passes.
"""

import torch
import ale_py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation


def test_environment_output():
    """Test that the environment outputs 210x160 grayscale frames."""
    print("=" * 70)
    print("TEST 1: Environment Output Shape")
    print("=" * 70)

    # Create environment
    env = gym.make("ALE/Pong-v5")
    env = GrayscaleObservation(env)

    # Reset and get initial observation
    obs, _ = env.reset()

    print(f"Environment output shape: {obs.shape}")
    print(f"  Expected: (210, 160)")
    print(f"  Data type: {obs.dtype}")
    print(f"  Value range: [{obs.min()}, {obs.max()}]")

    # Verify
    assert obs.shape == (210, 160), f"Expected (210, 160), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"

    print("✓ PASSED: Environment outputs correct shape\n")

    env.close()
    return obs


def test_tensor_conversion():
    """Test converting observation to tensor with correct dimensions."""
    print("=" * 70)
    print("TEST 2: Tensor Conversion")
    print("=" * 70)

    # Create a dummy observation
    obs = np.zeros((210, 160), dtype=np.uint8)

    print(f"Numpy array shape: {obs.shape}")

    # Convert to tensor the way it's done in ppo_pong.py
    # state_tensor = torch.from_numpy(self.state).unsqueeze(0).unsqueeze(0).float()

    # unsqueeze(0) adds a dimension at position 0
    step1 = torch.from_numpy(obs).unsqueeze(0)  # Add channel dim
    print(f"After first unsqueeze(0): {step1.shape}")
    assert step1.shape == (1, 210, 160), f"Expected (1, 210, 160), got {step1.shape}"

    step2 = step1.unsqueeze(0)  # Add batch dim
    print(f"After second unsqueeze(0): {step2.shape}")
    assert step2.shape == (
        1,
        1,
        210,
        160,
    ), f"Expected (1, 1, 210, 160), got {step2.shape}"

    step3 = step2.float()  # Convert to float
    print(f"After .float(): {step3.shape}, dtype: {step3.dtype}")

    print("✓ PASSED: Tensor conversion correct\n")


def test_cnn_feature_extractor():
    """Test CNN feature extractor with 210x160 input."""
    print("=" * 70)
    print("TEST 3: CNN Feature Extractor")
    print("=" * 70)

    from ppo_pong import CNNFeatureExtractor

    # Create extractor
    extractor = CNNFeatureExtractor(input_channels=1, feature_dim=512)

    # Create dummy input batch
    dummy_input = torch.randn(2, 1, 210, 160)  # batch_size=2

    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = extractor(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"  Expected: (2, 512)")

    assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"

    print("✓ PASSED: CNN feature extractor works with 210x160 input\n")


def test_actor_critic_network():
    """Test full actor-critic network."""
    print("=" * 70)
    print("TEST 4: Actor-Critic Network")
    print("=" * 70)

    from ppo_pong import PPOActorCritic

    # Create network
    net = PPOActorCritic(num_actions=3)  # 3 actions for Pong

    # Create dummy input
    dummy_input = torch.randn(4, 1, 210, 160)  # batch_size=4

    print(f"Input shape: {dummy_input.shape}")
    print(f"  Interpretation: (batch=4, channels=1, height=210, width=160)")

    # Forward pass
    with torch.no_grad():
        action_logits, values = net(dummy_input)

    print(f"\nOutputs:")
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"    Expected: (4, 3)")
    print(f"  Values shape: {values.shape}")
    print(f"    Expected: (4, 1)")

    assert action_logits.shape == (4, 3), f"Expected (4, 3), got {action_logits.shape}"
    assert values.shape == (4, 1), f"Expected (4, 1), got {values.shape}"

    print("✓ PASSED: Actor-critic network works correctly\n")


def test_get_action_and_value():
    """Test sampling actions with the network."""
    print("=" * 70)
    print("TEST 5: Get Action and Value")
    print("=" * 70)

    from ppo_pong import PPOActorCritic

    net = PPOActorCritic(num_actions=3)
    dummy_input = torch.randn(2, 1, 210, 160)

    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        action, log_prob, entropy, value = net.get_action_and_value(dummy_input)

    print(f"Action shape: {action.shape}")
    print(f"  Values: {action}")
    print(f"Log probability shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Value shape: {value.shape}")

    assert action.shape == (2,), f"Expected (2,), got {action.shape}"
    assert log_prob.shape == (2,), f"Expected (2,), got {log_prob.shape}"
    assert entropy.shape == (2,), f"Expected (2,), got {entropy.shape}"
    assert value.shape == (2, 1), f"Expected (2, 1), got {value.shape}"

    print("✓ PASSED: Action sampling works correctly\n")


def test_dimensions_match():
    """Test that all dimension calculations are correct."""
    print("=" * 70)
    print("TEST 6: Dimension Calculations")
    print("=" * 70)

    print("CNN Feature Extractor dimension calculations:")
    print("\nInput: 1x210x160")

    # Conv1: kernel=5, stride=2
    h1 = (210 - 5) // 2 + 1
    w1 = (160 - 5) // 2 + 1
    print(f"After Conv1 (k=5, s=2): 16x{h1}x{w1}")
    assert h1 == 103 and w1 == 78, f"Conv1 should be 103x78, got {h1}x{w1}"

    # Conv2: kernel=5, stride=2
    h2 = (h1 - 5) // 2 + 1
    w2 = (w1 - 5) // 2 + 1
    print(f"After Conv2 (k=5, s=2): 32x{h2}x{w2}")
    assert h2 == 50 and w2 == 37, f"Conv2 should be 50x37, got {h2}x{w2}"

    # Conv3: kernel=3, stride=1
    h3 = (h2 - 3) // 1 + 1
    w3 = (w2 - 3) // 1 + 1
    print(f"After Conv3 (k=3, s=1): 64x{h3}x{w3}")
    assert h3 == 48 and w3 == 35, f"Conv3 should be 48x35, got {h3}x{w3}"

    # Final flatten
    flattened = 64 * h3 * w3
    print(f"Flattened: {flattened} features")
    assert flattened == 107520, f"Should have 107520 features, got {flattened}"

    print(f"FC layer: {flattened} → 512 features")

    print("✓ PASSED: All dimensions calculated correctly\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  VERIFICATION: 210×160 Grayscale Modifications".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        # Run all tests
        test_environment_output()
        test_tensor_conversion()
        test_cnn_feature_extractor()
        test_actor_critic_network()
        test_get_action_and_value()
        test_dimensions_match()

        # Success!
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe 210×160 grayscale modifications are working correctly.")
        print("\nYou can now run:")
        print("  python ppo_pong.py                    # Single test run")
        print("  python train.py --num_seeds 3         # All experiments")
        print("  python utils.py                       # Analysis")
        print("\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
