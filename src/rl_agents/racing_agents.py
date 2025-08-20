import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Dict, Any, Type
import gymnasium as gym

class RacingFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for racing environment
    Separates car state, track state, and setup parameters
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions based on observation space
        total_obs_dim = observation_space.shape[0]
        
        # Assume fixed structure: car(12) + track(8) + lookahead(20) + setup(8 if enabled)
        self.car_dim = 12
        self.track_dim = 8
        self.lookahead_dim = 20
        self.setup_dim = total_obs_dim - (self.car_dim + self.track_dim + self.lookahead_dim)
        
        # Car state network
        self.car_net = nn.Sequential(
            nn.Linear(self.car_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Track state network
        self.track_net = nn.Sequential(
            nn.Linear(self.track_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Lookahead network (processes future track information)
        self.lookahead_net = nn.Sequential(
            nn.Linear(self.lookahead_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Setup network (if setup optimization is enabled)
        if self.setup_dim > 0:
            self.setup_net = nn.Sequential(
                nn.Linear(self.setup_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            combined_dim = 64 + 32 + 32 + 16
        else:
            self.setup_net = None
            combined_dim = 64 + 32 + 32
        
        # Combined feature network
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into different components
        car_obs = observations[:, :self.car_dim]
        track_obs = observations[:, self.car_dim:self.car_dim + self.track_dim]
        lookahead_obs = observations[:, self.car_dim + self.track_dim:self.car_dim + self.track_dim + self.lookahead_dim]
        
        # Process each component
        car_features = self.car_net(car_obs)
        track_features = self.track_net(track_obs)
        lookahead_features = self.lookahead_net(lookahead_obs)
        
        # Combine features
        combined_features = torch.cat([car_features, track_features, lookahead_features], dim=1)
        
        # Add setup features if available
        if self.setup_net is not None and self.setup_dim > 0:
            setup_obs = observations[:, -self.setup_dim:]
            setup_features = self.setup_net(setup_obs)
            combined_features = torch.cat([combined_features, setup_features], dim=1)
        
        return self.combined_net(combined_features)

class RacingActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for racing with specialized feature extraction
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                         features_extractor_class=RacingFeatureExtractor,
                         features_extractor_kwargs=dict(features_dim=256))

class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning - gradually increases difficulty
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.difficulty_level = 0
        self.performance_window = []
        self.window_size = 100
        self.difficulty_threshold = 0.7  # Performance threshold to increase difficulty
        
    def _on_step(self) -> bool:
        # Get episode info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.performance_window.append(episode_reward)
                
                if len(self.performance_window) > self.window_size:
                    self.performance_window.pop(0)
                
                # Check if we should increase difficulty
                if len(self.performance_window) == self.window_size:
                    avg_performance = np.mean(self.performance_window)
                    
                    # Normalize performance (environment-specific)
                    normalized_performance = min(1.0, max(0.0, (avg_performance + 50) / 100))
                    
                    if normalized_performance > self.difficulty_threshold and self.difficulty_level < 5:
                        self.difficulty_level += 1
                        self.performance_window = []  # Reset window
                        
                        # Update environment difficulty
                        if hasattr(self.training_env, 'set_difficulty'):
                            self.training_env.set_difficulty(self.difficulty_level)
                        
                        if self.verbose > 0:
                            print(f"Curriculum: Increased difficulty to level {self.difficulty_level}")
        
        return True

class PerformanceCallback(BaseCallback):
    """
    Callback to track and log racing-specific performance metrics
    """
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_lap_time = float('inf')
        self.lap_times = []
        self.crash_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_performance()
        
        # Track episode-level metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'best_lap_time' in info and info['best_lap_time'] < float('inf'):
                self.lap_times.append(info['best_lap_time'])
                if info['best_lap_time'] < self.best_lap_time:
                    self.best_lap_time = info['best_lap_time']
            
            if 'crashed' in info and info['crashed']:
                self.crash_count += 1
            
            if 'episode' in info:
                self.total_episodes += 1
        
        return True
    
    def _evaluate_performance(self):
        """Evaluate and log racing performance"""
        if len(self.lap_times) > 0:
            recent_lap_times = self.lap_times[-10:]  # Last 10 laps
            avg_lap_time = np.mean(recent_lap_times)
            std_lap_time = np.std(recent_lap_times)
            
            # Log performance metrics
            self.logger.record("racing/best_lap_time", self.best_lap_time)
            self.logger.record("racing/avg_lap_time", avg_lap_time)
            self.logger.record("racing/lap_time_std", std_lap_time)
            
        if self.total_episodes > 0:
            crash_rate = self.crash_count / self.total_episodes
            self.logger.record("racing/crash_rate", crash_rate)
        
        self.logger.record("racing/total_laps", len(self.lap_times))

class PPORacingAgent:
    """
    PPO agent specialized for racing
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = "auto"):
        
        self.env = env
        
        # PPO-specific hyperparameters optimized for racing
        self.model = PPO(
            policy=RacingActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            verbose=1,
            tensorboard_log="./logs/ppo_racing/"
        )
        
        # Callbacks
        self.callbacks = [
            CurriculumCallback(verbose=1),
            PerformanceCallback(eval_freq=1000, verbose=1)
        ]
    
    def train(self, total_timesteps: int = 100000):
        """Train the PPO agent"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            progress_bar=True
        )
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)
    
    def predict(self, observation, deterministic: bool = True):
        """Make a prediction"""
        return self.model.predict(observation, deterministic=deterministic)

class SACRacingAgent:
    """
    SAC agent specialized for racing
    """
    
    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 ent_coef: str = "auto",
                 target_update_interval: int = 1,
                 device: str = "auto"):
        
        self.env = env
        
        # SAC-specific hyperparameters optimized for racing
        self.model = SAC(
            policy="MlpPolicy",  # Using standard MLP for SAC
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            device=device,
            verbose=1,
            tensorboard_log="./logs/sac_racing/"
        )
        
        # Callbacks
        self.callbacks = [
            CurriculumCallback(verbose=1),
            PerformanceCallback(eval_freq=1000, verbose=1)
        ]
    
    def train(self, total_timesteps: int = 100000):
        """Train the SAC agent"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            progress_bar=True
        )
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = SAC.load(path, env=self.env)
    
    def predict(self, observation, deterministic: bool = True):
        """Make a prediction"""
        return self.model.predict(observation, deterministic=deterministic)

class AgentComparison:
    """
    Utility class for comparing different RL agents
    """
    
    def __init__(self, env_factory):
        self.env_factory = env_factory
        self.results = {}
    
    def train_and_evaluate_ppo(self, total_timesteps: int = 50000, n_eval_episodes: int = 10):
        """Train and evaluate PPO agent"""
        env = self.env_factory()
        agent = PPORacingAgent(env)
        
        print("Training PPO agent...")
        agent.train(total_timesteps)
        
        # Evaluate
        env = self.env_factory()
        lap_times, crash_count = self._evaluate_agent(agent, env, n_eval_episodes)
        
        self.results['PPO'] = {
            'lap_times': lap_times,
            'avg_lap_time': np.mean(lap_times) if lap_times else float('inf'),
            'best_lap_time': min(lap_times) if lap_times else float('inf'),
            'crash_rate': crash_count / n_eval_episodes,
            'consistency': np.std(lap_times) if len(lap_times) > 1 else float('inf')
        }
        
        # Save model
        agent.save("models/ppo_racing_agent")
        
        return self.results['PPO']
    
    def train_and_evaluate_sac(self, total_timesteps: int = 50000, n_eval_episodes: int = 10):
        """Train and evaluate SAC agent"""
        env = self.env_factory()
        agent = SACRacingAgent(env)
        
        print("Training SAC agent...")
        agent.train(total_timesteps)
        
        # Evaluate
        env = self.env_factory()
        lap_times, crash_count = self._evaluate_agent(agent, env, n_eval_episodes)
        
        self.results['SAC'] = {
            'lap_times': lap_times,
            'avg_lap_time': np.mean(lap_times) if lap_times else float('inf'),
            'best_lap_time': min(lap_times) if lap_times else float('inf'),
            'crash_rate': crash_count / n_eval_episodes,
            'consistency': np.std(lap_times) if len(lap_times) > 1 else float('inf')
        }
        
        # Save model
        agent.save("models/sac_racing_agent")
        
        return self.results['SAC']
    
    def _evaluate_agent(self, agent, env, n_episodes: int):
        """Evaluate an agent for n episodes"""
        lap_times = []
        crash_count = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_crashes = 0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info.get('crashed', False):
                    episode_crashes += 1
            
            # Record results
            if info.get('best_lap_time', float('inf')) < float('inf'):
                lap_times.append(info['best_lap_time'])
            
            if episode_crashes > 0:
                crash_count += 1
            
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Best lap: {info.get('best_lap_time', 'N/A'):.2f}s, "
                  f"Crashes: {episode_crashes}")
        
        return lap_times, crash_count
    
    def compare_results(self):
        """Compare results between different agents"""
        if not self.results:
            print("No results to compare. Run training first.")
            return
        
        print("\n" + "="*50)
        print("AGENT COMPARISON RESULTS")
        print("="*50)
        
        for agent_name, results in self.results.items():
            print(f"\n{agent_name}:")
            print(f"  Average lap time: {results['avg_lap_time']:.2f}s")
            print(f"  Best lap time: {results['best_lap_time']:.2f}s")
            print(f"  Consistency (std): {results['consistency']:.2f}s")
            print(f"  Crash rate: {results['crash_rate']:.1%}")
        
        # Determine winner
        if len(self.results) > 1:
            best_agent = min(self.results.keys(), 
                           key=lambda x: self.results[x]['avg_lap_time'])
            print(f"\nBest performing agent: {best_agent}")
        
        return self.results
