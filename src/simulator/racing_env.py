import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Dict, Tuple, Any, Optional, List
import pygame
import time

from .car_physics import RacingCarPhysics, CarSetup, CarState
from .track import RaceTrack, TrackGenerator

class RacingEnvironment(gym.Env):
    """
    Gymnasium environment for AI racing car training
    Combines car setup optimization with driving control
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 track_name: str = "oval",
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 5000,
                 enable_setup_optimization: bool = True,
                 safety_enabled: bool = True):
        
        super().__init__()
        
        self.track_name = track_name
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.enable_setup_optimization = enable_setup_optimization
        self.safety_enabled = safety_enabled
        
        # Create track
        self.track = self._create_track(track_name)
        
        # Initialize car setup and physics
        self.car_setup = CarSetup()
        self.car_physics = RacingCarPhysics(self.car_setup)
        self.car_state = CarState()
        
        # Environment state
        self.current_step = 0
        self.episode_start_time = 0.0
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.sector_times = [0.0, 0.0, 0.0]
        self.last_sector = 0
        self.lap_count = 0
        self.crashed = False
        self.off_track_time = 0.0
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Rendering
        self.screen = None
        self.clock = None
        self.screen_width = 1200
        self.screen_height = 800
        
        # Track progress for rewards
        self.last_track_distance = 0.0
        self.distance_traveled = 0.0
        self.wrong_way_counter = 0
        
    def _create_track(self, track_name: str) -> RaceTrack:
        """Create the specified track"""
        if track_name == "oval":
            track = RaceTrack("Test Oval")
            track.create_oval_track(1500.0, 300.0, 15.0)
        elif track_name == "monaco":
            track = RaceTrack("Monaco Inspired")
            track.create_monaco_inspired_track()
        elif track_name == "figure8":
            track = TrackGenerator.create_figure_eight()
        else:
            # Default to oval
            track = RaceTrack("Default Oval")
            track.create_oval_track(1200.0, 250.0, 12.0)
        
        return track
    
    def _setup_spaces(self):
        """Define action and observation spaces"""
        
        # Observation space
        # Car state: position, velocity, orientation, gear, rpm, tire info
        # Track state: curvature ahead, distance to track boundaries, sector info
        # Setup parameters (if optimization enabled)
        
        car_obs_dim = 12  # x, y, vx, vy, heading, angular_vel, speed, slip_angle, gear, rpm, tire_temp, tire_wear
        track_obs_dim = 8  # track_distance, lateral_offset, curvature, track_width, grip_level, next_curvature, elevation, heading_error
        lookahead_dim = 20  # curvature and grip for next 10 points ahead (10 * 2 = 20)
        
        obs_dim = car_obs_dim + track_obs_dim + lookahead_dim  # 12 + 8 + 20 = 40
        
        if self.enable_setup_optimization:
            setup_obs_dim = 8  # setup parameters
            obs_dim += setup_obs_dim  # 40 + 8 = 48
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space
        if self.enable_setup_optimization:
            # Driving actions: throttle, brake, steering
            # Setup actions: downforce_front, downforce_rear, tire_grip, weight_distribution, brake_balance
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.4]),  # steering, throttle, brake, setup params
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.7, 0.8]),
                dtype=np.float32
            )
        else:
            # Only driving actions
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),  # steering, throttle, brake
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset car state to starting position
        start_point = self.track.get_track_info_at_distance(0.0)
        self.car_state = CarState(
            x=start_point.x,
            y=start_point.y, 
            heading=start_point.heading,
            velocity_x=0.0,
            velocity_y=0.0,
            angular_velocity=0.0,
            gear=1,
            rpm=1000.0,
            tire_temp=80.0,
            tire_wear=0.0,
            track_position=0.0,
            lateral_offset=0.0
        )
        
        # Reset environment state
        self.current_step = 0
        self.episode_start_time = time.time()
        self.current_lap_time = 0.0
        self.sector_times = [0.0, 0.0, 0.0]
        self.last_sector = 0
        self.lap_count = 0
        self.crashed = False
        self.off_track_time = 0.0
        self.last_track_distance = 0.0
        self.distance_traveled = 0.0
        self.wrong_way_counter = 0
        
        # Randomize setup if optimization is enabled
        if self.enable_setup_optimization:
            self._randomize_setup()
        
        # Update physics with new setup
        self.car_physics = RacingCarPhysics(self.car_setup)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _randomize_setup(self):
        """Randomize car setup for domain randomization"""
        self.car_setup.downforce_front = self.np_random.uniform(0.2, 0.8)
        self.car_setup.downforce_rear = self.np_random.uniform(0.2, 0.8)
        self.car_setup.tire_grip = self.np_random.uniform(0.8, 1.2)
        self.car_setup.weight_distribution = self.np_random.uniform(0.4, 0.6)
        self.car_setup.brake_balance = self.np_random.uniform(0.5, 0.7)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        
        # Parse actions
        if self.enable_setup_optimization:
            steering = action[0]
            throttle = action[1] 
            brake = action[2]
            
            # Update setup parameters (with some smoothing to prevent rapid changes)
            alpha = 0.1  # Smoothing factor
            self.car_setup.downforce_front = (1-alpha) * self.car_setup.downforce_front + alpha * action[3]
            self.car_setup.downforce_rear = (1-alpha) * self.car_setup.downforce_rear + alpha * action[4]
            self.car_setup.tire_grip = (1-alpha) * self.car_setup.tire_grip + alpha * action[5]
            self.car_setup.weight_distribution = (1-alpha) * self.car_setup.weight_distribution + alpha * action[6]
            self.car_setup.brake_balance = (1-alpha) * self.car_setup.brake_balance + alpha * action[7]
        else:
            steering = action[0]
            throttle = action[1]
            brake = action[2]
        
        # Apply safety constraints if enabled
        if self.safety_enabled:
            steering, throttle, brake = self._apply_safety_constraints(steering, throttle, brake)
        
        # Update physics
        dt = 1.0 / 60.0  # 60 Hz simulation
        self.car_state = self.car_physics.update_physics(
            self.car_state, throttle, brake, steering, dt
        )
        
        # Update track position
        self._update_track_position()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Update step counter
        self.current_step += 1
        self.current_lap_time += dt
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_safety_constraints(self, steering: float, throttle: float, brake: float) -> Tuple[float, float, float]:
        """Apply safety constraints to prevent dangerous actions"""
        
        # Get track info at current position
        track_info = self.track.get_track_info_at_distance(self.car_state.track_position)
        
        # Speed-dependent steering limits
        max_lateral_accel = self.car_physics.get_max_lateral_acceleration(self.car_state.speed)
        if self.car_state.speed > 1.0:  # Avoid division by zero
            max_curvature = max_lateral_accel / (self.car_state.speed ** 2)
            max_steering = min(abs(steering), max_curvature * 2.0)  # Some safety margin
            steering = math.copysign(max_steering, steering)
        
        # Reduce throttle if approaching track boundaries
        if abs(self.car_state.lateral_offset) > track_info.track_width * 0.3:
            throttle *= 0.5
        
        # Emergency braking if way off track
        if abs(self.car_state.lateral_offset) > track_info.track_width * 0.8:
            brake = max(brake, 0.8)
            throttle = 0.0
        
        return steering, throttle, brake
    
    def _update_track_position(self):
        """Update car's position relative to track"""
        distance, lateral_offset = self.track.get_closest_point_on_track(
            self.car_state.x, self.car_state.y
        )
        
        # Handle track wrapping
        if distance < self.last_track_distance - self.track.track_length / 2:
            distance += self.track.track_length
            self.lap_count += 1
            
            # Record lap time
            if self.lap_count > 0:
                if self.current_lap_time < self.best_lap_time:
                    self.best_lap_time = self.current_lap_time
                print(f"Lap {self.lap_count} completed in {self.current_lap_time:.2f}s")
                self.current_lap_time = 0.0
        
        # Check for wrong way driving
        distance_delta = distance - self.last_track_distance
        if distance_delta < -self.track.track_length / 4:  # Allow for some wrapping
            self.wrong_way_counter += 1
        else:
            self.wrong_way_counter = max(0, self.wrong_way_counter - 1)
        
        self.distance_traveled += abs(distance_delta)
        self.last_track_distance = distance
        
        self.car_state.track_position = distance % self.track.track_length
        self.car_state.lateral_offset = lateral_offset
        
        # Check if off track
        track_info = self.track.get_track_info_at_distance(self.car_state.track_position)
        if abs(lateral_offset) > track_info.track_width / 2:
            self.off_track_time += 1.0 / 60.0
        else:
            self.off_track_time = 0.0
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        
        # Progress reward - encourage forward movement
        track_info = self.track.get_track_info_at_distance(self.car_state.track_position)
        
        # Speed reward (normalized by track characteristics)
        target_speed = min(50.0, 1.0 / max(abs(track_info.curvature), 0.01))  # Slower in turns
        speed_reward = self.car_state.speed / max(target_speed, 1.0)
        reward += speed_reward * 0.1
        
        # Track position reward
        if abs(self.car_state.lateral_offset) < track_info.track_width / 2:
            # On track
            center_reward = 1.0 - (abs(self.car_state.lateral_offset) / (track_info.track_width / 2))
            reward += center_reward * 0.1
        else:
            # Off track penalty
            reward -= 1.0
        
        # Progress reward
        distance_delta = self.car_state.track_position - self.last_track_distance
        if distance_delta > 0:  # Forward progress
            reward += distance_delta / self.track.track_length * 10.0
        
        # Lap completion bonus
        if self.lap_count > 0 and self.current_lap_time > 0:
            lap_time_bonus = max(0, 120.0 - self.current_lap_time) / 120.0  # Bonus for fast laps
            reward += lap_time_bonus * 5.0
        
        # Penalties
        if self.crashed:
            reward -= 10.0
        
        if self.wrong_way_counter > 30:  # Driving wrong way
            reward -= 5.0
        
        if self.off_track_time > 2.0:  # Too long off track
            reward -= 2.0
        
        # Smoothness reward (penalize excessive control changes)
        # This would require storing previous actions - simplified for now
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Crash detection (very high lateral acceleration or off track too long)
        if self.off_track_time > 5.0:
            self.crashed = True
            return True
        
        # Wrong way driving for too long
        if self.wrong_way_counter > 180:  # 3 seconds at 60 Hz
            return True
        
        # Completed multiple laps successfully
        if self.lap_count >= 3:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        # Car state
        car_obs = np.array([
            self.car_state.x / 1000.0,  # Normalize positions
            self.car_state.y / 1000.0,
            self.car_state.velocity_x / 100.0,  # Normalize velocities
            self.car_state.velocity_y / 100.0,
            self.car_state.heading / math.pi,  # Normalize angle
            self.car_state.angular_velocity / 10.0,
            self.car_state.speed / 100.0,  # Normalize speed
            self.car_state.slip_angle / math.pi,
            self.car_state.gear / 8.0,  # Normalize gear
            self.car_state.rpm / 15000.0,  # Normalize RPM
            self.car_state.tire_temp / 150.0,  # Normalize temperature
            self.car_state.tire_wear  # Already 0-1
        ], dtype=np.float32)
        
        # Track state
        track_info = self.track.get_track_info_at_distance(self.car_state.track_position)
        next_info = self.track.get_track_info_at_distance(
            self.car_state.track_position + self.car_state.speed * 2.0  # 2 seconds ahead
        )
        
        heading_error = self.car_state.heading - track_info.heading
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
        
        track_obs = np.array([
            self.car_state.track_position / self.track.track_length,
            self.car_state.lateral_offset / track_info.track_width,
            track_info.curvature * 100.0,  # Scale curvature
            track_info.track_width / 20.0,  # Normalize width
            track_info.grip_level,
            next_info.curvature * 100.0,
            track_info.elevation / 100.0,  # Normalize elevation
            heading_error / math.pi
        ], dtype=np.float32)
        
        # Lookahead information
        lookahead_obs = []
        for i in range(1, 11):  # Next 10 points
            future_distance = self.car_state.track_position + i * 50.0  # Every 50m
            future_info = self.track.get_track_info_at_distance(future_distance)
            lookahead_obs.extend([
                future_info.curvature * 100.0,
                future_info.grip_level
            ])
        lookahead_obs = np.array(lookahead_obs, dtype=np.float32)
        
        # Combine observations
        observation = np.concatenate([car_obs, track_obs, lookahead_obs])
        
        # Add setup parameters if optimization is enabled
        if self.enable_setup_optimization:
            setup_obs = np.array([
                self.car_setup.downforce_front,
                self.car_setup.downforce_rear,
                self.car_setup.tire_grip,
                self.car_setup.weight_distribution,
                self.car_setup.brake_balance,
                self.car_setup.drag_coefficient,
                self.car_setup.tire_wear_rate,
                self.car_setup.final_drive / 5.0  # Normalize
            ], dtype=np.float32)
            observation = np.concatenate([observation, setup_obs])
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'lap_count': self.lap_count,
            'current_lap_time': self.current_lap_time,
            'best_lap_time': self.best_lap_time,
            'speed': self.car_state.speed,
            'gear': self.car_state.gear,
            'rpm': self.car_state.rpm,
            'tire_temp': self.car_state.tire_temp,
            'tire_wear': self.car_state.tire_wear,
            'track_position': self.car_state.track_position,
            'lateral_offset': self.car_state.lateral_offset,
            'off_track_time': self.off_track_time,
            'crashed': self.crashed,
            'distance_traveled': self.distance_traveled,
            'setup': {
                'downforce_front': self.car_setup.downforce_front,
                'downforce_rear': self.car_setup.downforce_rear,
                'tire_grip': self.car_setup.tire_grip,
                'weight_distribution': self.car_setup.weight_distribution,
                'brake_balance': self.car_setup.brake_balance
            }
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((0, 100, 0))  # Green background
        
        # Calculate view parameters
        zoom = 0.5
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Draw track
        self._draw_track(zoom, center_x, center_y)
        
        # Draw car
        self._draw_car(zoom, center_x, center_y)
        
        # Draw UI
        self._draw_ui()
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _render_rgb_array(self):
        """Render as RGB array"""
        # Similar to human rendering but return pixel array
        self._render_human()
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
    
    def _draw_track(self, zoom: float, center_x: int, center_y: int):
        """Draw the race track"""
        if len(self.track.centerline) < 2:
            return
        
        # Draw track boundaries
        left_points = []
        right_points = []
        center_points = []
        
        for point in self.track.centerline:
            # Convert world coordinates to screen coordinates
            screen_x = center_x + (point.x - self.car_state.x) * zoom
            screen_y = center_y - (point.y - self.car_state.y) * zoom
            
            # Calculate track boundary points
            perpendicular = point.heading + math.pi/2
            half_width = point.track_width / 2 * zoom
            
            left_x = screen_x + math.cos(perpendicular) * half_width
            left_y = screen_y - math.sin(perpendicular) * half_width
            right_x = screen_x - math.cos(perpendicular) * half_width
            right_y = screen_y + math.sin(perpendicular) * half_width
            
            left_points.append((left_x, left_y))
            right_points.append((right_x, right_y))
            center_points.append((screen_x, screen_y))
        
        # Draw track surface
        if len(left_points) > 2:
            track_points = left_points + list(reversed(right_points))
            pygame.draw.polygon(self.screen, (80, 80, 80), track_points)
        
        # Draw centerline
        if len(center_points) > 1:
            pygame.draw.lines(self.screen, (255, 255, 255), False, center_points, 2)
    
    def _draw_car(self, zoom: float, center_x: int, center_y: int):
        """Draw the racing car"""
        car_length = 5.0 * zoom
        car_width = 2.0 * zoom
        
        # Car corners in car coordinate system
        corners = [
            (-car_length/2, -car_width/2),
            (car_length/2, -car_width/2),
            (car_length/2, car_width/2),
            (-car_length/2, car_width/2)
        ]
        
        # Rotate and translate to world coordinates
        cos_h = math.cos(self.car_state.heading)
        sin_h = math.sin(self.car_state.heading)
        
        screen_corners = []
        for dx, dy in corners:
            # Rotate
            world_x = dx * cos_h - dy * sin_h
            world_y = dx * sin_h + dy * cos_h
            
            # Translate to screen
            screen_x = center_x + world_x
            screen_y = center_y - world_y
            screen_corners.append((screen_x, screen_y))
        
        # Draw car body
        pygame.draw.polygon(self.screen, (255, 0, 0), screen_corners)
        
        # Draw direction indicator
        front_x = center_x + math.cos(self.car_state.heading) * car_length/2
        front_y = center_y - math.sin(self.car_state.heading) * car_length/2
        pygame.draw.circle(self.screen, (255, 255, 0), (int(front_x), int(front_y)), 3)
    
    def _draw_ui(self):
        """Draw user interface elements"""
        font = pygame.font.Font(None, 36)
        
        # Speed
        speed_text = font.render(f"Speed: {self.car_state.speed:.1f} km/h", True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 10))
        
        # Gear and RPM
        gear_text = font.render(f"Gear: {self.car_state.gear}  RPM: {self.car_state.rpm:.0f}", True, (255, 255, 255))
        self.screen.blit(gear_text, (10, 50))
        
        # Lap info
        lap_text = font.render(f"Lap: {self.lap_count}  Time: {self.current_lap_time:.2f}s", True, (255, 255, 255))
        self.screen.blit(lap_text, (10, 90))
        
        # Best lap
        if self.best_lap_time < float('inf'):
            best_text = font.render(f"Best: {self.best_lap_time:.2f}s", True, (255, 255, 255))
            self.screen.blit(best_text, (10, 130))
        
        # Track position
        pos_text = font.render(f"Track pos: {self.car_state.track_position:.0f}m", True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 170))
    
    def close(self):
        """Clean up rendering"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
