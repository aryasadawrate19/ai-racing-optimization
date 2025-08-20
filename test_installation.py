#!/usr/bin/env python3
"""
Test script to verify the AI Racing System installation and functionality
"""

import sys
import os
import traceback

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"✗ Gymnasium import failed: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"✓ Stable-Baselines3 {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"✗ Stable-Baselines3 import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import pygame
        print(f"✓ Pygame {pygame.__version__}")
    except ImportError as e:
        print(f"✗ Pygame import failed: {e}")
        return False
    
    print("All core imports successful!\n")
    return True

def test_fastf1():
    """Test FastF1 import (optional for basic functionality)"""
    print("Testing FastF1 (optional)...")
    try:
        import fastf1
        print(f"✓ FastF1 {fastf1.__version__}")
        return True
    except ImportError as e:
        print(f"⚠ FastF1 not available: {e}")
        print("  F1 data validation will be limited")
        return False

def test_simulator():
    """Test the racing simulator components"""
    print("Testing simulator components...")
    
    try:
        from simulator.car_physics import RacingCarPhysics, CarSetup, CarState
        print("✓ Car physics module")
        
        # Test basic car setup
        setup = CarSetup()
        physics = RacingCarPhysics(setup)
        state = CarState()
        
        # Test physics update
        new_state = physics.update_physics(state, 0.5, 0.0, 0.1, 0.016)
        print("✓ Physics simulation")
        
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from simulator.track import RaceTrack
        print("✓ Track module")
        
        # Test track creation
        track = RaceTrack("Test Track")
        track.create_oval_track()
        print("✓ Track creation")
        
    except Exception as e:
        print(f"✗ Track test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from simulator.racing_env import RacingEnvironment
        print("✓ Racing environment")
        
        # Test environment creation
        env = RacingEnvironment(track_name="oval", max_episode_steps=100)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step")
        env.close()
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        traceback.print_exc()
        return False
    
    print("Simulator tests successful!\n")
    return True

def test_safety_system():
    """Test the safety system"""
    print("Testing safety system...")
    
    try:
        from safety.safety_shield import SafetyLimits, SafetyShield, ControlBarrierFunction
        from simulator.car_physics import CarState
        from simulator.track import RaceTrack
        print("✓ Safety imports")
        
        # Test safety components
        limits = SafetyLimits()
        shield = SafetyShield(limits)
        cbf = ControlBarrierFunction(limits)
        
        # Test with dummy data
        state = CarState()
        track = RaceTrack("Test")
        track.create_oval_track()
        
        barriers = cbf.evaluate_all_barriers(state, track)
        is_safe = cbf.is_safe(state, track)
        print("✓ Safety evaluation")
        
    except Exception as e:
        print(f"✗ Safety system test failed: {e}")
        traceback.print_exc()
        return False
    
    print("Safety system tests successful!\n")
    return True

def test_rl_agents():
    """Test RL agent components"""
    print("Testing RL agents...")
    
    try:
        from rl_agents.racing_agents import PPORacingAgent, SACRacingAgent
        from simulator.racing_env import RacingEnvironment
        print("✓ RL agent imports")
        
        # Create simple environment for testing
        env = RacingEnvironment(track_name="oval", max_episode_steps=50)
        
        # Test PPO agent creation
        ppo_agent = PPORacingAgent(env)
        print("✓ PPO agent creation")
        
        # Test SAC agent creation  
        sac_agent = SACRacingAgent(env)
        print("✓ SAC agent creation")
        
        env.close()
        
    except Exception as e:
        print(f"✗ RL agents test failed: {e}")
        traceback.print_exc()
        return False
    
    print("RL agents tests successful!\n")
    return True

def test_quick_training():
    """Test a very quick training run"""
    print("Testing quick training run...")
    
    try:
        from simulator.racing_env import RacingEnvironment
        from rl_agents.racing_agents import PPORacingAgent
        
        # Create environment
        env = RacingEnvironment(track_name="oval", max_episode_steps=100)
        
        # Create agent
        agent = PPORacingAgent(env)
        
        # Very short training
        print("Starting minimal training (this may take a minute)...")
        agent.train(total_timesteps=512)  # Minimal training
        print("✓ Training completed")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        print("✓ Prediction test")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        traceback.print_exc()
        return False
    
    print("Quick training test successful!\n")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("AI RACING SYSTEM - INSTALLATION TEST")
    print("="*60)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Core Imports", test_imports),
        ("FastF1 (Optional)", test_fastf1),
        ("Simulator", test_simulator),
        ("Safety System", test_safety_system),
        ("RL Agents", test_rl_agents),
        ("Quick Training", test_quick_training)
    ]
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            if not result and test_name != "FastF1 (Optional)":
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            traceback.print_exc()
            if test_name != "FastF1 (Optional)":
                all_passed = False
        print()
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The AI Racing System is ready to use!")
        print("\nTry running the demo:")
        print('python main.py --mode demo')
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and install missing dependencies.")
    print("="*60)

if __name__ == "__main__":
    main()
