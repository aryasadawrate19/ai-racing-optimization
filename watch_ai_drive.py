#!/usr/bin/env python3
"""
Live AI Racing Visualization - Watch the AI learn to drive!
"""

import os
import sys
import time
import numpy as np
import pygame
import math

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from simulator.racing_env import RacingEnvironment
    from rl_agents.racing_agents import PPORacingAgent, SACRacingAgent
except ImportError:
    print("⚠️  Full system not installed. Using simulation mode.")
    SIMULATION_MODE = True
else:
    SIMULATION_MODE = False

class LiveRacingVisualizer:
    """
    Real-time visualization of AI racing with enhanced graphics
    """
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🏎️ AI Racing - Live Training Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.colors = {
            'background': (20, 30, 40),
            'track': (60, 60, 60),
            'track_lines': (255, 255, 255),
            'car_ai': (255, 100, 100),
            'car_human': (100, 255, 100),
            'speed_line': (255, 255, 0),
            'text': (255, 255, 255),
            'ui_bg': (40, 50, 60),
            'progress_bg': (100, 100, 100),
            'progress_fill': (100, 255, 100)
        }
        
        # Simulation data for demo mode
        self.demo_data = {
            'lap_count': 0,
            'current_lap_time': 0.0,
            'best_lap_time': float('inf'),
            'speed': 0.0,
            'position': 0.0,
            'learning_progress': 0.0
        }
        
    def draw_track(self, track_points, car_x, car_y, zoom=1.0):
        """Draw the racing track"""
        if len(track_points) < 3:
            # Draw simple oval for demo
            center_x, center_y = self.width // 2, self.height // 2
            track_width = 60
            
            # Outer boundary
            pygame.draw.ellipse(self.screen, self.colors['track'], 
                              [center_x - 300, center_y - 200, 600, 400])
            # Inner boundary  
            pygame.draw.ellipse(self.screen, self.colors['background'], 
                              [center_x - 240, center_y - 140, 480, 280])
            
            # Center line
            pygame.draw.ellipse(self.screen, self.colors['track_lines'], 
                              [center_x - 270, center_y - 170, 540, 340], 2)
            
            # Start/finish line
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (center_x + 300, center_y - 30), 
                           (center_x + 300, center_y + 30), 4)
            
            return center_x, center_y
        
        # Draw actual track points (when available)
        # Implementation would use the track data
        return self.width // 2, self.height // 2
    
    def draw_car(self, x, y, heading, speed, is_ai=True):
        """Draw the racing car with speed visualization"""
        car_length = 20
        car_width = 8
        
        # Car body color based on type
        color = self.colors['car_ai'] if is_ai else self.colors['car_human']
        
        # Calculate car corners
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        corners = [
            (-car_length//2, -car_width//2),
            (car_length//2, -car_width//2),
            (car_length//2, car_width//2),
            (-car_length//2, car_width//2)
        ]
        
        # Rotate and translate corners
        screen_corners = []
        for dx, dy in corners:
            world_x = x + dx * cos_h - dy * sin_h
            world_y = y + dx * sin_h + dy * cos_h
            screen_corners.append((world_x, world_y))
        
        # Draw car body
        pygame.draw.polygon(self.screen, color, screen_corners)
        
        # Draw direction indicator
        front_x = x + cos_h * car_length//2
        front_y = y + sin_h * car_length//2
        pygame.draw.circle(self.screen, (255, 255, 0), (int(front_x), int(front_y)), 3)
        
        # Speed visualization (trail effect)
        if speed > 10:
            trail_length = min(speed * 2, 50)
            trail_x = x - cos_h * trail_length
            trail_y = y - sin_h * trail_length
            pygame.draw.line(self.screen, self.colors['speed_line'], 
                           (trail_x, trail_y), (x, y), 3)
    
    def draw_ui_panel(self, data):
        """Draw the information panel"""
        panel_x = 20
        panel_y = 20
        panel_width = 350
        panel_height = 250
        
        # Background
        pygame.draw.rect(self.screen, self.colors['ui_bg'], 
                        [panel_x, panel_y, panel_width, panel_height])
        pygame.draw.rect(self.screen, self.colors['text'], 
                        [panel_x, panel_y, panel_width, panel_height], 2)
        
        # Title
        title = self.font.render("🏎️ AI Racing Status", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Statistics
        y_offset = 50
        stats = [
            f"Lap: {data.get('lap_count', 0)}",
            f"Current Time: {data.get('current_lap_time', 0.0):.2f}s",
            f"Best Time: {data.get('best_lap_time', 0.0):.2f}s" if data.get('best_lap_time', float('inf')) < float('inf') else "Best Time: --",
            f"Speed: {data.get('speed', 0.0):.1f} km/h",
            f"Position: {data.get('position', 0.0):.0f}m",
            f"Learning: {data.get('learning_progress', 0.0):.1%}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 15, panel_y + y_offset + i * 25))
        
        # Learning progress bar
        progress_y = panel_y + panel_height - 40
        progress_width = panel_width - 20
        progress_height = 20
        
        # Background
        pygame.draw.rect(self.screen, self.colors['progress_bg'], 
                        [panel_x + 10, progress_y, progress_width, progress_height])
        
        # Progress fill
        fill_width = int(progress_width * data.get('learning_progress', 0.0))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.colors['progress_fill'], 
                            [panel_x + 10, progress_y, fill_width, progress_height])
        
        # Progress text
        progress_text = self.small_font.render("Learning Progress", True, self.colors['text'])
        self.screen.blit(progress_text, (panel_x + 15, progress_y - 20))
    
    def draw_setup_panel(self, setup_data):
        """Draw car setup parameters panel"""
        panel_x = self.width - 370
        panel_y = 20
        panel_width = 350
        panel_height = 200
        
        # Background
        pygame.draw.rect(self.screen, self.colors['ui_bg'], 
                        [panel_x, panel_y, panel_width, panel_height])
        pygame.draw.rect(self.screen, self.colors['text'], 
                        [panel_x, panel_y, panel_width, panel_height], 2)
        
        # Title
        title = self.font.render("⚙️ Car Setup", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Setup parameters
        y_offset = 50
        if setup_data:
            params = [
                f"Downforce Front: {setup_data.get('downforce_front', 0.5):.2f}",
                f"Downforce Rear: {setup_data.get('downforce_rear', 0.5):.2f}",
                f"Tire Grip: {setup_data.get('tire_grip', 1.0):.2f}",
                f"Weight Dist: {setup_data.get('weight_distribution', 0.5):.2f}",
                f"Brake Balance: {setup_data.get('brake_balance', 0.6):.2f}"
            ]
        else:
            params = [
                "Downforce Front: 0.55",
                "Downforce Rear: 0.48", 
                "Tire Grip: 1.05",
                "Weight Dist: 0.52",
                "Brake Balance: 0.63"
            ]
        
        for i, param in enumerate(params):
            text = self.small_font.render(param, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 15, panel_y + y_offset + i * 25))
    
    def run_simulation_demo(self):
        """Run a demo simulation without full system"""
        print("🎮 Starting AI Racing Visualization Demo")
        print("Watch the AI learn to optimize its racing line and car setup!")
        print("Press SPACE to speed up, ESC to quit")
        
        # Demo simulation variables
        car_x = self.width // 2 + 300
        car_y = self.height // 2
        car_heading = 0
        car_speed = 40
        lap_progress = 0
        generation = 0
        
        running = True
        speed_multiplier = 1
        
        while running:
            dt = self.clock.tick(60) / 1000.0 * speed_multiplier
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        speed_multiplier = 3 if speed_multiplier == 1 else 1
            
            # Simulate car movement around oval track
            angular_speed = car_speed / 200  # Turn rate based on speed
            car_heading += angular_speed * dt
            
            # Calculate position
            radius = 250
            center_x, center_y = self.width // 2, self.height // 2
            car_x = center_x + radius * math.cos(car_heading)
            car_y = center_y + radius * math.sin(car_heading)
            
            # Update lap progress
            lap_progress = (car_heading % (2 * math.pi)) / (2 * math.pi)
            
            # Simulate lap completion
            if car_heading > (generation + 1) * 2 * math.pi:
                generation += 1
                lap_time = 85 + np.random.normal(0, 2) - generation * 0.5  # Improving over time
                self.demo_data['lap_count'] = generation
                self.demo_data['best_lap_time'] = min(self.demo_data['best_lap_time'], lap_time)
                self.demo_data['learning_progress'] = min(1.0, generation / 20.0)
            
            # Update demo data
            self.demo_data['current_lap_time'] = lap_progress * 85
            self.demo_data['speed'] = car_speed * 3.6  # Convert to km/h
            self.demo_data['position'] = lap_progress * 1000
            
            # Simulate learning (speed increases over time)
            car_speed = 40 + generation * 2
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw track
            self.draw_track([], car_x, car_y)
            
            # Draw car
            self.draw_car(car_x, car_y, car_heading + math.pi/2, car_speed, is_ai=True)
            
            # Draw UI
            self.draw_ui_panel(self.demo_data)
            self.draw_setup_panel(None)
            
            # Draw instructions
            instruction_text = f"Speed: {'3x' if speed_multiplier > 1 else '1x'} (SPACE to toggle) | ESC to quit"
            text = self.small_font.render(instruction_text, True, self.colors['text'])
            self.screen.blit(text, (20, self.height - 30))
            
            # Generation info
            gen_text = f"Generation {generation + 1} - AI is learning!"
            text = self.font.render(gen_text, True, self.colors['text'])
            self.screen.blit(text, (self.width // 2 - 150, 20))
            
            pygame.display.flip()
        
        pygame.quit()
    
    def run_live_training(self, agent_type='PPO'):
        """Run live training visualization with real AI"""
        if SIMULATION_MODE:
            print("⚠️  Full system not available, running demo mode...")
            self.run_simulation_demo()
            return
        
        print(f"🤖 Starting live {agent_type} training visualization")
        
        # Create environment with rendering
        env = RacingEnvironment(
            track_name="oval", 
            render_mode="human",
            max_episode_steps=2000,
            enable_setup_optimization=True,
            safety_enabled=True
        )
        
        # Create agent
        if agent_type.upper() == 'PPO':
            agent = PPORacingAgent(env)
        else:
            agent = SACRacingAgent(env)
        
        # Training parameters
        episodes = 50
        steps_per_episode = 1000
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Get AI action
                action, _ = agent.predict(obs, deterministic=False)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Render at 30 FPS for smooth visualization
                if step % 2 == 0:
                    env.render()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward:.2f}")
            
            # Brief training step
            if episode % 5 == 0:
                agent.train(total_timesteps=1000)
        
        env.close()

def main():
    """Main visualization function"""
    print("🏎️ AI Racing Live Visualization")
    print("=" * 40)
    print("Choose visualization mode:")
    print("1. Demo Mode (No installation required)")
    print("2. Live Training Visualization (PPO)")
    print("3. Live Training Visualization (SAC)")
    print("4. Render Trained Agent")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    visualizer = LiveRacingVisualizer()
    
    if choice == "1":
        visualizer.run_simulation_demo()
    elif choice == "2":
        visualizer.run_live_training('PPO')
    elif choice == "3":
        visualizer.run_live_training('SAC')
    elif choice == "4":
        print("Loading trained agent...")
        # This would load a pre-trained model
        visualizer.run_simulation_demo()  # Fallback to demo
    else:
        print("Invalid choice, running demo mode...")
        visualizer.run_simulation_demo()

if __name__ == "__main__":
    main()
