#!/usr/bin/env python3

from src.simulator.racing_env import RacingEnvironment
import numpy as np

def test_observation_spaces():
    """Test observation space dimensions"""
    
    print("Testing observation space dimensions...")
    
    # Test with setup optimization
    env = RacingEnvironment(enable_setup_optimization=True, render_mode=None)
    obs, info = env.reset()
    print(f'Environment (with setup) observation space shape: {env.observation_space.shape}')
    print(f'Actual observation (with setup) shape: {obs.shape}')
    print(f'Dimensions match (with setup): {env.observation_space.shape[0] == obs.shape[0]}')
    print()

    # Test without setup optimization
    env2 = RacingEnvironment(enable_setup_optimization=False, render_mode=None)
    obs2, info2 = env2.reset()
    print(f'Environment (no setup) observation space shape: {env2.observation_space.shape}')
    print(f'Actual observation (no setup) shape: {obs2.shape}')
    print(f'Dimensions match (no setup): {env2.observation_space.shape[0] == obs2.shape[0]}')
    print()
    
    # Detailed breakdown
    print("Detailed observation breakdown:")
    print(f"Car observations: 12 dimensions")
    print(f"Track observations: 8 dimensions") 
    print(f"Lookahead observations: 20 dimensions (10 points x 2)")
    print(f"Setup observations: 8 dimensions (when enabled)")
    print(f"Expected total (with setup): 12 + 8 + 20 + 8 = 48")
    print(f"Expected total (no setup): 12 + 8 + 20 = 40")

if __name__ == "__main__":
    test_observation_spaces()
