#!/usr/bin/env python3

print("Testing AI Racing Visualization System...")

try:
    # Test imports
    from src.simulator.racing_env import RacingEnvironment
    from src.rl_agents.racing_agents import PPORacingAgent
    from src.simulator.car_physics import RacingCarPhysics, CarSetup, CarState
    print("✅ All imports successful")
    
    # Test environment creation
    env = RacingEnvironment(enable_setup_optimization=True, render_mode=None)
    obs, info = env.reset()
    print(f"✅ Environment created - Obs shape: {obs.shape}")
    
    # Test a single step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment step successful - Speed: {info['speed']:.2f} m/s")
    
    # Test agent creation
    agent = PPORacingAgent(env)
    print("✅ PPO agent created successfully")
    
    # Test agent prediction
    action, _ = agent.predict(obs, deterministic=False)
    print(f"✅ Agent prediction successful - Action shape: {action.shape}")
    
    print("\n🎉 All components working correctly!")
    print("The visualization system is ready to use.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
