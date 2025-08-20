import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass

from ..simulator.car_physics import CarState, CarSetup
from ..simulator.track import RaceTrack

@dataclass
class SafetyLimits:
    """Safety limits for various car parameters"""
    max_lateral_accel: float = 2.5  # Maximum lateral acceleration (g)
    max_longitudinal_accel: float = 1.5  # Maximum longitudinal acceleration (g)
    max_speed: float = 100.0  # Maximum speed (m/s)
    min_following_distance: float = 20.0  # Minimum following distance (m)
    track_boundary_margin: float = 1.0  # Safety margin from track boundaries (m)
    max_slip_angle: float = 0.3  # Maximum slip angle (radians)
    max_yaw_rate: float = 2.0  # Maximum yaw rate (rad/s)

class ControlBarrierFunction:
    """
    Control Barrier Function for racing safety
    Ensures the car stays within safe operating limits
    """
    
    def __init__(self, safety_limits: SafetyLimits):
        self.limits = safety_limits
        self.g = 9.81  # Gravitational acceleration
        
    def barrier_lateral_acceleration(self, state: CarState, track: RaceTrack) -> float:
        """
        Barrier function for lateral acceleration limits
        
        Args:
            state: Current car state
            track: Race track
            
        Returns:
            Barrier function value (positive = safe, negative = unsafe)
        """
        # Get track information
        track_info = track.get_track_info_at_distance(state.track_position)
        
        # Calculate maximum safe lateral acceleration based on speed and curvature
        if state.speed > 1.0:
            required_lateral_accel = abs(track_info.curvature) * state.speed**2
            max_safe_lateral_accel = self.limits.max_lateral_accel * self.g
            
            # Barrier function: h(x) = max_accel - required_accel
            barrier_value = max_safe_lateral_accel - required_lateral_accel
            return barrier_value
        else:
            return float('inf')  # Safe at low speeds
    
    def barrier_track_boundaries(self, state: CarState, track: RaceTrack) -> float:
        """
        Barrier function for track boundary constraints
        
        Args:
            state: Current car state
            track: Race track
            
        Returns:
            Barrier function value
        """
        track_info = track.get_track_info_at_distance(state.track_position)
        
        # Distance to track boundaries
        max_offset = track_info.track_width / 2 - self.limits.track_boundary_margin
        current_offset = abs(state.lateral_offset)
        
        # Barrier function: h(x) = max_offset - current_offset
        barrier_value = max_offset - current_offset
        return barrier_value
    
    def barrier_speed_limit(self, state: CarState) -> float:
        """
        Barrier function for maximum speed constraint
        
        Args:
            state: Current car state
            
        Returns:
            Barrier function value
        """
        barrier_value = self.limits.max_speed - state.speed
        return barrier_value
    
    def barrier_slip_angle(self, state: CarState) -> float:
        """
        Barrier function for slip angle constraint
        
        Args:
            state: Current car state
            
        Returns:
            Barrier function value
        """
        barrier_value = self.limits.max_slip_angle - abs(state.slip_angle)
        return barrier_value
    
    def barrier_yaw_rate(self, state: CarState) -> float:
        """
        Barrier function for yaw rate constraint
        
        Args:
            state: Current car state
            
        Returns:
            Barrier function value
        """
        barrier_value = self.limits.max_yaw_rate - abs(state.angular_velocity)
        return barrier_value
    
    def evaluate_all_barriers(self, state: CarState, track: RaceTrack) -> dict:
        """
        Evaluate all barrier functions
        
        Args:
            state: Current car state
            track: Race track
            
        Returns:
            Dictionary of barrier function values
        """
        barriers = {
            'lateral_accel': self.barrier_lateral_acceleration(state, track),
            'track_boundaries': self.barrier_track_boundaries(state, track),
            'speed_limit': self.barrier_speed_limit(state),
            'slip_angle': self.barrier_slip_angle(state),
            'yaw_rate': self.barrier_yaw_rate(state)
        }
        return barriers
    
    def is_safe(self, state: CarState, track: RaceTrack, margin: float = 0.1) -> bool:
        """
        Check if current state is safe
        
        Args:
            state: Current car state
            track: Race track
            margin: Safety margin
            
        Returns:
            True if state is safe
        """
        barriers = self.evaluate_all_barriers(state, track)
        return all(value > margin for value in barriers.values())

class SafetyShield:
    """
    Safety shield that modifies unsafe actions to ensure safety
    """
    
    def __init__(self, safety_limits: SafetyLimits):
        self.limits = safety_limits
        self.cbf = ControlBarrierFunction(safety_limits)
        self.alpha = 1.0  # Class-K function parameter
        
    def apply_safety_filter(self, 
                           state: CarState, 
                           track: RaceTrack,
                           desired_action: np.ndarray,
                           setup: CarSetup) -> Tuple[np.ndarray, bool]:
        """
        Apply safety filter to desired action
        
        Args:
            state: Current car state
            track: Race track
            desired_action: Desired control action [steering, throttle, brake]
            setup: Car setup parameters
            
        Returns:
            Tuple of (safe_action, was_modified)
        """
        steering, throttle, brake = desired_action[0], desired_action[1], desired_action[2]
        was_modified = False
        
        # Get current barrier values
        barriers = self.cbf.evaluate_all_barriers(state, track)
        
        # Check lateral acceleration constraint
        if barriers['lateral_accel'] < 0.5:  # Close to unsafe
            # Reduce speed if going too fast for the turn
            track_info = track.get_track_info_at_distance(state.track_position)
            if abs(track_info.curvature) > 0.01 and state.speed > 10.0:
                # Calculate safe speed for this corner
                max_lat_accel = self.limits.max_lateral_accel * self.g * 0.8  # Safety factor
                safe_speed = math.sqrt(max_lat_accel / abs(track_info.curvature))
                
                if state.speed > safe_speed:
                    # Apply braking
                    brake = max(brake, 0.5)
                    throttle = min(throttle, 0.2)
                    was_modified = True
        
        # Check track boundary constraint
        if barriers['track_boundaries'] < 1.0:  # Getting close to boundary
            track_info = track.get_track_info_at_distance(state.track_position)
            
            # Reduce lateral input if heading towards boundary
            if state.lateral_offset * steering > 0:  # Steering away from center
                steering_reduction = 1.0 - min(1.0, abs(barriers['track_boundaries']))
                steering *= steering_reduction
                was_modified = True
            
            # Emergency correction if very close to boundary
            if abs(barriers['track_boundaries']) < 0.5:
                # Counter-steer to get back to center
                correction_strength = 0.5 - abs(barriers['track_boundaries'])
                steering = -math.copysign(correction_strength, state.lateral_offset)
                throttle = min(throttle, 0.3)
                was_modified = True
        
        # Check speed limit
        if barriers['speed_limit'] < 5.0:  # Within 5 m/s of limit
            if state.speed > self.limits.max_speed * 0.95:
                throttle = 0.0
                brake = max(brake, 0.3)
                was_modified = True
        
        # Check slip angle
        if barriers['slip_angle'] < 0.1:  # High slip angle
            # Reduce throttle and steering inputs
            throttle *= 0.5
            steering *= 0.7
            was_modified = True
        
        # Check yaw rate
        if barriers['yaw_rate'] < 0.5:  # High yaw rate
            # Reduce steering input
            steering *= 0.5
            throttle *= 0.8
            was_modified = True
        
        # Emergency stop if multiple barriers are violated
        unsafe_barriers = sum(1 for value in barriers.values() if value < 0)
        if unsafe_barriers >= 2:
            throttle = 0.0
            brake = 1.0
            steering *= 0.3
            was_modified = True
        
        safe_action = np.array([steering, throttle, brake])
        return safe_action, was_modified
    
    def get_safety_intervention_level(self, state: CarState, track: RaceTrack) -> float:
        """
        Get the level of safety intervention needed
        
        Args:
            state: Current car state
            track: Race track
            
        Returns:
            Intervention level (0 = no intervention, 1 = maximum intervention)
        """
        barriers = self.cbf.evaluate_all_barriers(state, track)
        
        # Calculate minimum barrier value
        min_barrier = min(barriers.values())
        
        # Convert to intervention level
        if min_barrier > 1.0:
            return 0.0  # No intervention needed
        elif min_barrier < 0.0:
            return 1.0  # Maximum intervention
        else:
            return 1.0 - min_barrier  # Gradual intervention

class RacingViabilityKernel:
    """
    Viability kernel for racing - defines the set of states from which
    safe racing is possible
    """
    
    def __init__(self, safety_limits: SafetyLimits):
        self.limits = safety_limits
        self.cbf = ControlBarrierFunction(safety_limits)
    
    def is_viable_state(self, state: CarState, track: RaceTrack, 
                       horizon: float = 2.0) -> bool:
        """
        Check if a state is viable (can be kept safe for the given horizon)
        
        Args:
            state: Current car state
            track: Race track
            horizon: Time horizon for viability check (seconds)
            
        Returns:
            True if state is viable
        """
        # Current safety check
        if not self.cbf.is_safe(state, track):
            return False
        
        # Predict future states with simple forward simulation
        dt = 0.1
        steps = int(horizon / dt)
        current_state = state
        
        for _ in range(steps):
            # Simple forward prediction (constant velocity model)
            next_state = CarState(
                x=current_state.x + current_state.velocity_x * dt,
                y=current_state.y + current_state.velocity_y * dt,
                heading=current_state.heading + current_state.angular_velocity * dt,
                velocity_x=current_state.velocity_x,
                velocity_y=current_state.velocity_y,
                angular_velocity=current_state.angular_velocity,
                speed=current_state.speed,
                slip_angle=current_state.slip_angle,
                gear=current_state.gear,
                rpm=current_state.rpm,
                tire_temp=current_state.tire_temp,
                tire_wear=current_state.tire_wear,
                track_position=current_state.track_position + current_state.speed * dt,
                lateral_offset=current_state.lateral_offset
            )
            
            # Update track position
            distance, lateral_offset = track.get_closest_point_on_track(
                next_state.x, next_state.y
            )
            next_state.track_position = distance % track.track_length
            next_state.lateral_offset = lateral_offset
            
            # Check if predicted state is safe
            if not self.cbf.is_safe(next_state, track, margin=0.5):
                return False
            
            current_state = next_state
        
        return True
    
    def compute_viability_margin(self, state: CarState, track: RaceTrack) -> float:
        """
        Compute how far the state is from the viability boundary
        
        Args:
            state: Current car state
            track: Race track
            
        Returns:
            Viability margin (positive = viable, negative = not viable)
        """
        barriers = self.cbf.evaluate_all_barriers(state, track)
        return min(barriers.values())

class AdaptiveSafetySystem:
    """
    Adaptive safety system that learns from racing incidents
    """
    
    def __init__(self, initial_limits: SafetyLimits):
        self.limits = initial_limits
        self.shield = SafetyShield(initial_limits)
        self.viability = RacingViabilityKernel(initial_limits)
        
        # Learning parameters
        self.incident_history = []
        self.intervention_history = []
        self.performance_history = []
        
    def record_incident(self, state: CarState, track: RaceTrack, 
                       incident_type: str, severity: float):
        """
        Record a safety incident for learning
        
        Args:
            state: State when incident occurred
            track: Race track
            incident_type: Type of incident ('off_track', 'spin', 'collision', etc.)
            severity: Severity of incident (0-1)
        """
        incident = {
            'state': state,
            'track_position': state.track_position,
            'speed': state.speed,
            'lateral_offset': state.lateral_offset,
            'type': incident_type,
            'severity': severity,
            'barriers': self.shield.cbf.evaluate_all_barriers(state, track)
        }
        self.incident_history.append(incident)
        
        # Adapt safety limits based on incidents
        self._adapt_safety_limits()
    
    def _adapt_safety_limits(self):
        """Adapt safety limits based on incident history"""
        if len(self.incident_history) < 5:
            return
        
        recent_incidents = self.incident_history[-10:]  # Last 10 incidents
        
        # Count incident types
        off_track_incidents = sum(1 for inc in recent_incidents if inc['type'] == 'off_track')
        speed_incidents = sum(1 for inc in recent_incidents if inc['type'] == 'overspeed')
        
        # Adapt limits
        if off_track_incidents > 3:
            # Increase track boundary margin
            self.limits.track_boundary_margin *= 1.1
            
        if speed_incidents > 2:
            # Reduce maximum speed
            self.limits.max_speed *= 0.95
        
        # Update safety components
        self.shield = SafetyShield(self.limits)
        self.viability = RacingViabilityKernel(self.limits)
    
    def get_safety_statistics(self) -> dict:
        """Get safety system performance statistics"""
        total_interventions = len(self.intervention_history)
        total_incidents = len(self.incident_history)
        
        if total_interventions > 0:
            intervention_rate = total_incidents / total_interventions
        else:
            intervention_rate = 0.0
        
        return {
            'total_interventions': total_interventions,
            'total_incidents': total_incidents,
            'intervention_effectiveness': 1.0 - intervention_rate,
            'current_limits': {
                'max_lateral_accel': self.limits.max_lateral_accel,
                'max_speed': self.limits.max_speed,
                'track_boundary_margin': self.limits.track_boundary_margin
            }
        }
