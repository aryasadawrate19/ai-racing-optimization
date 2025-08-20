#!/usr/bin/env python3

from src.simulator.racing_env import RacingEnvironment
import numpy as np

def test_physics():
    """Test the car physics to see if the slip variables are working"""
    
    print("Testing car physics...")
    
    # Create environment 
    env = RacingEnvironment(enable_setup_optimization=True, render_mode=None)
    obs, info = env.reset()
    
    # Take a few steps to test physics
    for i in range(5):
        # Random action: steering, throttle, brake, setup params
        action = np.random.uniform(-1, 1, size=env.action_space.shape[0])
        action[1] = abs(action[1])  # Throttle should be positive
        action[2] = abs(action[2])  # Brake should be positive
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Success - Speed: {info['speed']:.2f} m/s")
        except Exception as e:
            print(f"Step {i+1}: Error - {e}")
            break
    
    print("Physics test completed.")

if __name__ == "__main__":
    test_physics()
