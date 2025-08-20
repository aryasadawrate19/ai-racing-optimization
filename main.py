#!/usr/bin/env python3
"""
Main training script for AI-Powered Racing Car Setup and Lap Time Optimization
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulator.racing_env import RacingEnvironment
from rl_agents.racing_agents import PPORacingAgent, SACRacingAgent, AgentComparison
from data_analysis.f1_data_analyzer import F1DataAnalyzer, create_ai_vs_f1_comparison
from safety.safety_shield import SafetyLimits, AdaptiveSafetySystem

def create_environment(track_name: str = "oval", 
                      enable_setup_optimization: bool = True,
                      safety_enabled: bool = True,
                      render_mode: str = None) -> RacingEnvironment:
    """Create racing environment with specified parameters"""
    env = RacingEnvironment(
        track_name=track_name,
        render_mode=render_mode,
        max_episode_steps=5000,
        enable_setup_optimization=enable_setup_optimization,
        safety_enabled=safety_enabled
    )
    return env

def train_single_agent(agent_type: str, 
                      track_name: str = "oval",
                      timesteps: int = 50000,
                      enable_setup_optimization: bool = True,
                      safety_enabled: bool = True,
                      render_mode: str = None) -> dict:
    """
    Train a single RL agent
    
    Args:
        agent_type: 'PPO' or 'SAC'
        track_name: Name of the track to train on
        timesteps: Number of training timesteps
        enable_setup_optimization: Whether to enable car setup optimization
        safety_enabled: Whether to enable safety constraints
        
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training {agent_type} Agent")
    print(f"Track: {track_name}")
    print(f"Setup Optimization: {enable_setup_optimization}")
    print(f"Safety Enabled: {safety_enabled}")
    print(f"Timesteps: {timesteps:,}")
    print(f"{'='*60}")
    
    # Create environment
    env = create_environment(
        track_name=track_name,
        enable_setup_optimization=enable_setup_optimization,
        safety_enabled=safety_enabled,
        render_mode=render_mode
    )
    
    # Create agent
    if agent_type.upper() == 'PPO':
        agent = PPORacingAgent(env)
    elif agent_type.upper() == 'SAC':
        agent = SACRacingAgent(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Train agent
    print(f"\nStarting {agent_type} training...")
    start_time = datetime.now()
    
    try:
        agent.train(total_timesteps=timesteps)
        training_time = datetime.now() - start_time
        print(f"Training completed in {training_time}")
        
        # Save model
        model_path = f"models/{agent_type.lower()}_racing_{track_name}_{timesteps}"
        os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate agent
        print(f"\nEvaluating {agent_type} agent...")
        evaluation_results = evaluate_agent(agent, env, n_episodes=10)
        
        results = {
            'agent_type': agent_type,
            'track_name': track_name,
            'training_timesteps': timesteps,
            'training_time': training_time.total_seconds(),
            'setup_optimization': enable_setup_optimization,
            'safety_enabled': safety_enabled,
            'model_path': model_path,
            **evaluation_results
        }
        
        return results
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        return {'error': 'Training interrupted'}
    except Exception as e:
        print(f"\nError during training: {e}")
        return {'error': str(e)}

def evaluate_agent(agent, env, n_episodes: int = 10) -> dict:
    """
    Evaluate an agent's performance
    
    Args:
        agent: Trained RL agent
        env: Racing environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results dictionary
    """
    lap_times = []
    crash_count = 0
    total_distance = 0
    setup_parameters = []
    
    print(f"Running {n_episodes} evaluation episodes...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_crashes = 0
        episode_distance = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info.get('crashed', False):
                episode_crashes += 1
            
            episode_distance = info.get('distance_traveled', 0)
        
        # Record results
        if info.get('best_lap_time', float('inf')) < float('inf'):
            lap_times.append(info['best_lap_time'])
        
        if episode_crashes > 0:
            crash_count += 1
        
        total_distance += episode_distance
        
        # Record setup parameters if available
        if 'setup' in info:
            setup_parameters.append(info['setup'])
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Best lap: {info.get('best_lap_time', 'N/A'):.2f}s, "
              f"Crashes: {episode_crashes}, "
              f"Distance: {episode_distance:.0f}m")
    
    # Calculate statistics
    results = {
        'lap_times': lap_times,
        'avg_lap_time': np.mean(lap_times) if lap_times else float('inf'),
        'best_lap_time': min(lap_times) if lap_times else float('inf'),
        'worst_lap_time': max(lap_times) if lap_times else float('inf'),
        'lap_time_std': np.std(lap_times) if len(lap_times) > 1 else 0.0,
        'crash_rate': crash_count / n_episodes,
        'completion_rate': len(lap_times) / n_episodes,
        'total_distance': total_distance,
        'avg_distance_per_episode': total_distance / n_episodes,
        'setup_parameters': setup_parameters
    }
    
    return results

def compare_agents(track_name: str = "oval", 
                  timesteps: int = 30000,
                  enable_setup_optimization: bool = True) -> dict:
    """
    Compare PPO and SAC agents on the same track
    
    Args:
        track_name: Track to train and compare on
        timesteps: Training timesteps for each agent
        enable_setup_optimization: Whether to enable setup optimization
        
    Returns:
        Comparison results
    """
    print(f"\n{'='*60}")
    print(f"AGENT COMPARISON")
    print(f"Track: {track_name}")
    print(f"Setup Optimization: {enable_setup_optimization}")
    print(f"Timesteps per agent: {timesteps:,}")
    print(f"{'='*60}")
    
    # Train and evaluate both agents
    ppo_results = train_single_agent(
        'PPO', track_name, timesteps, enable_setup_optimization
    )
    
    sac_results = train_single_agent(
        'SAC', track_name, timesteps, enable_setup_optimization
    )
    
    # Compare results
    comparison = {
        'track_name': track_name,
        'setup_optimization': enable_setup_optimization,
        'ppo_results': ppo_results,
        'sac_results': sac_results,
        'comparison': {}
    }
    
    if 'error' not in ppo_results and 'error' not in sac_results:
        # Determine winner
        ppo_time = ppo_results.get('best_lap_time', float('inf'))
        sac_time = sac_results.get('best_lap_time', float('inf'))
        
        comparison['comparison'] = {
            'winner': 'PPO' if ppo_time < sac_time else 'SAC',
            'time_difference': abs(ppo_time - sac_time),
            'ppo_crash_rate': ppo_results.get('crash_rate', 1.0),
            'sac_crash_rate': sac_results.get('crash_rate', 1.0),
            'ppo_consistency': ppo_results.get('lap_time_std', float('inf')),
            'sac_consistency': sac_results.get('lap_time_std', float('inf'))
        }
    
    return comparison

def validate_against_f1_data(agent_results: dict, track_name: str = "Monaco") -> dict:
    """
    Validate AI performance against real F1 data
    
    Args:
        agent_results: Results from AI agent evaluation
        track_name: F1 track name for comparison
        
    Returns:
        Validation results
    """
    print(f"\n{'='*60}")
    print(f"F1 DATA VALIDATION")
    print(f"Track: {track_name}")
    print(f"{'='*60}")
    
    try:
        # Create F1 data analyzer
        f1_analyzer = F1DataAnalyzer(year=2023)
        
        # Create comparison
        comparison = create_ai_vs_f1_comparison(
            agent_results, f1_analyzer, track_name
        )
        
        # Print results
        print(f"\nValidation Results:")
        print(f"Track: {comparison['track']}")
        print(f"Data Source: {comparison['data_source']}")
        print(f"F1 Reference Time: {comparison['f1_reference_time']:.2f}s")
        print(f"AI Best Time: {comparison['ai_best_time']:.2f}s")
        print(f"Performance Gap: {comparison['performance_gap']:.2f}s")
        
        if 'performance_percentage' in comparison:
            print(f"Performance Gap: {comparison['performance_percentage']:.1f}%")
        
        print(f"AI Completion Rate: {comparison['ai_completion_rate']:.1%}")
        
        return comparison
        
    except Exception as e:
        print(f"Error during F1 validation: {e}")
        return {'error': str(e)}

def demonstrate_safety_system(track_name: str = "oval") -> dict:
    """
    Demonstrate the safety system functionality
    
    Args:
        track_name: Track to demonstrate on
        
    Returns:
        Safety demonstration results
    """
    print(f"\n{'='*60}")
    print(f"SAFETY SYSTEM DEMONSTRATION")
    print(f"Track: {track_name}")
    print(f"{'='*60}")
    
    # Create environment with safety enabled
    env_safe = create_environment(track_name, safety_enabled=True)
    env_unsafe = create_environment(track_name, safety_enabled=False)
    
    # Train a basic agent on unsafe environment
    print("Training agent without safety constraints...")
    agent = PPORacingAgent(env_unsafe)
    agent.train(total_timesteps=10000)
    
    # Test with and without safety
    print("\nTesting with safety system...")
    safe_results = evaluate_agent(agent, env_safe, n_episodes=5)
    
    print("\nTesting without safety system...")
    unsafe_results = evaluate_agent(agent, env_unsafe, n_episodes=5)
    
    safety_comparison = {
        'track_name': track_name,
        'safe_results': safe_results,
        'unsafe_results': unsafe_results,
        'safety_improvement': {
            'crash_rate_reduction': unsafe_results['crash_rate'] - safe_results['crash_rate'],
            'completion_rate_improvement': safe_results['completion_rate'] - unsafe_results['completion_rate']
        }
    }
    
    print(f"\nSafety System Results:")
    print(f"Crash rate without safety: {unsafe_results['crash_rate']:.1%}")
    print(f"Crash rate with safety: {safe_results['crash_rate']:.1%}")
    print(f"Safety improvement: {safety_comparison['safety_improvement']['crash_rate_reduction']:.1%}")
    
    return safety_comparison

def create_performance_plots(results: dict, save_dir: str = "plots"):
    """
    Create performance visualization plots
    
    Args:
        results: Results dictionary from training/evaluation
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Lap time distribution
    if 'lap_times' in results and results['lap_times']:
        plt.figure(figsize=(10, 6))
        plt.hist(results['lap_times'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Lap Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Lap Time Distribution')
        plt.axvline(results['avg_lap_time'], color='red', linestyle='--', 
                   label=f'Average: {results["avg_lap_time"]:.2f}s')
        plt.axvline(results['best_lap_time'], color='green', linestyle='--',
                   label=f'Best: {results["best_lap_time"]:.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/lap_time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Setup parameters (if available)
    if 'setup_parameters' in results and results['setup_parameters']:
        setup_data = results['setup_parameters']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Downforce
        downforce_front = [s['downforce_front'] for s in setup_data]
        downforce_rear = [s['downforce_rear'] for s in setup_data]
        axes[0, 0].plot(downforce_front, label='Front')
        axes[0, 0].plot(downforce_rear, label='Rear')
        axes[0, 0].set_title('Downforce Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Tire grip
        tire_grip = [s['tire_grip'] for s in setup_data]
        axes[0, 1].plot(tire_grip)
        axes[0, 1].set_title('Tire Grip')
        axes[0, 1].grid(True)
        
        # Weight distribution
        weight_dist = [s['weight_distribution'] for s in setup_data]
        axes[1, 0].plot(weight_dist)
        axes[1, 0].set_title('Weight Distribution')
        axes[1, 0].grid(True)
        
        # Brake balance
        brake_balance = [s['brake_balance'] for s in setup_data]
        axes[1, 1].plot(brake_balance)
        axes[1, 1].set_title('Brake Balance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/setup_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Racing Car Training and Optimization')
    parser.add_argument('--mode', choices=['train', 'compare', 'validate', 'safety', 'demo'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--agent', choices=['PPO', 'SAC', 'both'], default='PPO', 
                       help='RL agent type')
    parser.add_argument('--track', default='oval', 
                       help='Track name (oval, monaco, figure8)')
    parser.add_argument('--timesteps', type=int, default=30000, 
                       help='Training timesteps')
    parser.add_argument('--no-setup', action='store_true', 
                       help='Disable setup optimization')
    parser.add_argument('--no-safety', action='store_true', 
                       help='Disable safety system')
    parser.add_argument('--render', action='store_true', 
                       help='Enable rendering during evaluation')
    parser.add_argument('--watch', action='store_true',
                       help='Enable live training visualization')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"AI-POWERED RACING CAR OPTIMIZATION")
    print(f"Mode: {args.mode}")
    print(f"Agent: {args.agent}")
    print(f"Track: {args.track}")
    print(f"{'='*60}")
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    enable_setup = not args.no_setup
    enable_safety = not args.no_safety
    render_mode = "human" if args.render or args.watch else None
    
    if args.mode == 'train':
        # Train single agent
        results = train_single_agent(
            args.agent, args.track, args.timesteps, 
            enable_setup, enable_safety, render_mode
        )
        
        if 'error' not in results:
            create_performance_plots(results)
            print(f"\nTraining completed successfully!")
            print(f"Best lap time: {results['best_lap_time']:.2f}s")
            print(f"Crash rate: {results['crash_rate']:.1%}")
        
    elif args.mode == 'compare':
        # Compare PPO and SAC
        comparison = compare_agents(args.track, args.timesteps, enable_setup)
        
        if 'comparison' in comparison and comparison['comparison']:
            comp = comparison['comparison']
            print(f"\nComparison Results:")
            print(f"Winner: {comp['winner']}")
            print(f"Time difference: {comp['time_difference']:.2f}s")
            print(f"PPO crash rate: {comp['ppo_crash_rate']:.1%}")
            print(f"SAC crash rate: {comp['sac_crash_rate']:.1%}")
        
    elif args.mode == 'validate':
        # Validate against F1 data
        # First need to train an agent
        results = train_single_agent(
            args.agent, args.track, args.timesteps, enable_setup, enable_safety
        )
        
        if 'error' not in results:
            validation = validate_against_f1_data(results, 'Monaco')
            
    elif args.mode == 'safety':
        # Demonstrate safety system
        safety_demo = demonstrate_safety_system(args.track)
        
    elif args.mode == 'demo':
        # Quick demonstration
        print("\nRunning quick demonstration...")
        
        # Train a small PPO agent
        demo_results = train_single_agent(
            'PPO', args.track, 10000, enable_setup, enable_safety
        )
        
        if 'error' not in demo_results:
            print(f"\nDemo Results:")
            print(f"Best lap time: {demo_results['best_lap_time']:.2f}s")
            print(f"Average lap time: {demo_results['avg_lap_time']:.2f}s")
            print(f"Crash rate: {demo_results['crash_rate']:.1%}")
            print(f"Completion rate: {demo_results['completion_rate']:.1%}")
            
            # Create plots
            create_performance_plots(demo_results)
            print(f"Performance plots saved to plots/")
        
        print(f"\nDemo completed! Check the models/ and plots/ directories for results.")
    
    print(f"\n{'='*60}")
    print(f"EXECUTION COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
