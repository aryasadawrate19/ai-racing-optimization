#!/usr/bin/env python3

import sys
import numpy as np

def test_core_system():
    """Test the core system without pygame dependencies"""
    
    print("🧪 Testing AI Racing Core System...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("📦 Testing imports...")
        from src.simulator.racing_env import RacingEnvironment
        from src.rl_agents.racing_agents import PPORacingAgent, SACRacingAgent
        from src.simulator.car_physics import RacingCarPhysics, CarSetup, CarState
        print("   ✅ All imports successful")
        
        # Test environment creation (without rendering)
        print("\n🏎️ Testing environment...")
        env = RacingEnvironment(enable_setup_optimization=True, render_mode=None)
        obs, info = env.reset()
        print(f"   ✅ Environment created - Obs shape: {obs.shape}")
        print(f"   ✅ Action space shape: {env.action_space.shape}")
        
        # Test physics
        print("\n⚙️ Testing physics...")
        for i in range(3):
            action = env.action_space.sample()
            action[1] = abs(action[1])  # Ensure positive throttle
            action[2] = abs(action[2])  # Ensure positive brake
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: Speed={info['speed']:.2f} m/s, Reward={reward:.3f}")
        
        # Test agent creation
        print("\n🤖 Testing PPO agent...")
        ppo_agent = PPORacingAgent(env)
        print("   ✅ PPO agent created")
        
        # Test agent prediction
        action, _ = ppo_agent.predict(obs, deterministic=False)
        print(f"   ✅ PPO prediction successful - Action: {action}")
        
        print("\n🤖 Testing SAC agent...")
        sac_agent = SACRacingAgent(env)
        print("   ✅ SAC agent created")
        
        action, _ = sac_agent.predict(obs, deterministic=False)
        print(f"   ✅ SAC prediction successful - Action: {action}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("Your AI Racing System is working perfectly!")
        print("\n🎮 Ready for visualization:")
        print("   python watch_ai_drive.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_system()
    sys.exit(0 if success else 1)
