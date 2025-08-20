import numpy as np
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CarSetup:
    """Car setup parameters that can be optimized"""
    # Aerodynamics
    downforce_front: float = 0.5  # Front downforce coefficient (0-1)
    downforce_rear: float = 0.5   # Rear downforce coefficient (0-1)
    drag_coefficient: float = 0.3  # Drag coefficient
    
    # Tires
    tire_grip: float = 1.0        # Tire grip multiplier (0.5-1.5)
    tire_wear_rate: float = 1.0   # How fast tires degrade
    
    # Mechanical
    weight_distribution: float = 0.5  # Weight distribution front/rear (0.3-0.7)
    brake_balance: float = 0.6        # Brake balance front/rear (0.4-0.8)
    
    # Gearing
    gear_ratios: np.ndarray = None    # Gear ratios array
    final_drive: float = 3.5          # Final drive ratio
    
    def __post_init__(self):
        if self.gear_ratios is None:
            # Default 8-speed F1 gearbox ratios
            self.gear_ratios = np.array([3.5, 2.8, 2.2, 1.8, 1.5, 1.3, 1.1, 1.0])

@dataclass
class CarState:
    """Current state of the racing car"""
    # Position and orientation
    x: float = 0.0              # X position (m)
    y: float = 0.0              # Y position (m)
    heading: float = 0.0        # Heading angle (radians)
    
    # Velocity
    velocity_x: float = 0.0     # Velocity in X direction (m/s)
    velocity_y: float = 0.0     # Velocity in Y direction (m/s)
    angular_velocity: float = 0.0  # Angular velocity (rad/s)
    
    # Car-relative velocity
    speed: float = 0.0          # Forward speed (m/s)
    slip_angle: float = 0.0     # Slip angle (radians)
    
    # Drivetrain
    gear: int = 1               # Current gear
    rpm: float = 1000.0         # Engine RPM
    
    # Tires
    tire_temp: float = 80.0     # Tire temperature (°C)
    tire_wear: float = 0.0      # Tire wear (0-1)
    
    # Track position
    track_position: float = 0.0  # Distance along track centerline (m)
    lateral_offset: float = 0.0  # Distance from track centerline (m)

class RacingCarPhysics:
    """
    2D racing car physics model based on bicycle model with F1-inspired parameters
    """
    
    def __init__(self, setup: CarSetup):
        self.setup = setup
        
        # Physical constants
        self.mass = 798.0  # F1 minimum weight (kg) including driver
        self.wheelbase = 3.7  # F1 wheelbase (m)
        self.track_width = 2.0  # F1 track width (m)
        self.frontal_area = 1.5  # Frontal area (m²)
        self.air_density = 1.225  # Air density at sea level (kg/m³)
        
        # Engine parameters
        self.max_power = 1000000  # Max power (W) - 1000 HP
        self.max_torque = 500     # Max torque (Nm)
        self.max_rpm = 15000      # Max RPM
        self.idle_rpm = 1000      # Idle RPM
        
        # Tire parameters (Pacejka Magic Formula)
        self.tire_b = 10.0        # Stiffness factor
        self.tire_c = 1.4         # Shape factor
        self.tire_d = 1.0         # Peak factor
        self.tire_e = -0.5        # Curvature factor
        
        # Aerodynamic reference values
        self.downforce_ref = 3000  # Reference downforce at reference speed (N)
        self.drag_ref = 800        # Reference drag at reference speed (N)
        self.ref_speed = 60.0      # Reference speed for aero forces (m/s)
        
    def pacejka_force(self, slip_ratio: float, normal_force: float, 
                     grip_multiplier: float = 1.0) -> float:
        """
        Calculate tire force using Pacejka Magic Formula
        
        Args:
            slip_ratio: Slip ratio (dimensionless)
            normal_force: Normal force on tire (N)
            grip_multiplier: Grip multiplier for different track conditions
            
        Returns:
            Tire force (N)
        """
        # Apply grip multiplier
        mu = self.tire_d * grip_multiplier
        
        # Pacejka formula: F = D * sin(C * arctan(B * slip - E * (B * slip - arctan(B * slip))))
        bx = self.tire_b * slip_ratio
        force = (mu * normal_force * 
                math.sin(self.tire_c * math.atan(bx - self.tire_e * (bx - math.atan(bx)))))
        
        return force
    
    def calculate_aerodynamic_forces(self, speed: float) -> Tuple[float, float, float]:
        """
        Calculate aerodynamic forces
        
        Args:
            speed: Current speed (m/s)
            
        Returns:
            Tuple of (drag_force, downforce_front, downforce_rear) in Newtons
        """
        # Dynamic pressure
        q = 0.5 * self.air_density * speed * speed
        
        # Drag force
        drag_force = q * self.setup.drag_coefficient * self.frontal_area
        
        # Downforce
        total_downforce = q * (self.setup.downforce_front + self.setup.downforce_rear) * self.frontal_area
        downforce_front = total_downforce * self.setup.downforce_front / (self.setup.downforce_front + self.setup.downforce_rear)
        downforce_rear = total_downforce - downforce_front
        
        return drag_force, downforce_front, downforce_rear
    
    def calculate_engine_force(self, throttle: float, rpm: float, gear: int) -> float:
        """
        Calculate engine force based on throttle, RPM, and gear
        
        Args:
            throttle: Throttle input (0-1)
            rpm: Engine RPM
            gear: Current gear (1-8)
            
        Returns:
            Engine force at wheels (N)
        """
        # Engine torque curve (simplified)
        if rpm < self.idle_rpm:
            rpm = self.idle_rpm
        elif rpm > self.max_rpm:
            rpm = self.max_rpm
            
        # Simplified torque curve - peak torque at mid-range RPM
        normalized_rpm = rpm / self.max_rpm
        torque_multiplier = 1.0 - 0.5 * (normalized_rpm - 0.6) ** 2
        torque_multiplier = max(0.1, torque_multiplier)
        
        engine_torque = self.max_torque * torque_multiplier * throttle
        
        # Convert to wheel force through gearing
        if 1 <= gear <= len(self.setup.gear_ratios):
            gear_ratio = self.setup.gear_ratios[gear - 1] * self.setup.final_drive
            wheel_force = engine_torque * gear_ratio / (self.track_width / 2)
        else:
            wheel_force = 0.0
            
        return wheel_force
    
    def update_physics(self, state: CarState, throttle: float, brake: float, 
                      steering: float, dt: float) -> CarState:
        """
        Update car physics for one time step
        
        Args:
            state: Current car state
            throttle: Throttle input (0-1)
            brake: Brake input (0-1)
            steering: Steering input (-1 to 1, radians)
            dt: Time step (seconds)
            
        Returns:
            Updated car state
        """
        new_state = CarState(
            x=state.x, y=state.y, heading=state.heading,
            velocity_x=state.velocity_x, velocity_y=state.velocity_y,
            angular_velocity=state.angular_velocity,
            gear=state.gear, rpm=state.rpm,
            tire_temp=state.tire_temp, tire_wear=state.tire_wear,
            track_position=state.track_position, lateral_offset=state.lateral_offset
        )
        
        # Calculate current speed and slip angle
        speed = math.sqrt(state.velocity_x**2 + state.velocity_y**2)
        new_state.speed = speed
        
        if speed > 0.1:
            new_state.slip_angle = math.atan2(state.velocity_y, state.velocity_x) - state.heading
        else:
            new_state.slip_angle = 0.0
        
        # Calculate aerodynamic forces
        drag_force, downforce_front, downforce_rear = self.calculate_aerodynamic_forces(speed)
        
        # Normal forces (weight + downforce)
        weight_front = self.mass * 9.81 * self.setup.weight_distribution
        weight_rear = self.mass * 9.81 * (1.0 - self.setup.weight_distribution)
        normal_front = weight_front + downforce_front
        normal_rear = weight_rear + downforce_rear
        
        # Engine force
        engine_force = self.calculate_engine_force(throttle, state.rpm, state.gear)
        
        # Braking force
        max_brake_front = normal_front * 0.8  # Max friction coefficient for brakes
        max_brake_rear = normal_rear * 0.8
        brake_force_front = brake * self.setup.brake_balance * max_brake_front
        brake_force_rear = brake * (1.0 - self.setup.brake_balance) * max_brake_rear
        total_brake_force = brake_force_front + brake_force_rear
        
        # Longitudinal force
        longitudinal_force = engine_force - total_brake_force - drag_force
        
        # Tire forces (simplified lateral dynamics)
        if speed > 0.1:
            # Lateral slip angle for cornering
            front_slip = steering - new_state.slip_angle
            rear_slip = -new_state.slip_angle
            
            # Lateral forces using Pacejka model
            lateral_front = self.pacejka_force(front_slip, normal_front, self.setup.tire_grip)
            lateral_rear = self.pacejka_force(rear_slip, normal_rear, self.setup.tire_grip)
        else:
            front_slip = 0.0
            rear_slip = 0.0
            lateral_front = 0.0
            lateral_rear = 0.0
        
        # Forces in car coordinate system
        force_x = longitudinal_force
        force_y = lateral_front + lateral_rear
        
        # Moment about center of mass
        moment = (lateral_front * self.wheelbase * self.setup.weight_distribution - 
                 lateral_rear * self.wheelbase * (1.0 - self.setup.weight_distribution))
        
        # Acceleration in car coordinates
        accel_x = force_x / self.mass
        accel_y = force_y / self.mass
        angular_accel = moment / (self.mass * self.wheelbase**2 / 12)  # Simplified moment of inertia
        
        # Transform to global coordinates
        cos_h = math.cos(state.heading)
        sin_h = math.sin(state.heading)
        
        accel_global_x = accel_x * cos_h - accel_y * sin_h
        accel_global_y = accel_x * sin_h + accel_y * cos_h
        
        # Update velocities
        new_state.velocity_x += accel_global_x * dt
        new_state.velocity_y += accel_global_y * dt
        new_state.angular_velocity += angular_accel * dt
        
        # Apply some damping to angular velocity for stability
        new_state.angular_velocity *= 0.95
        
        # Update position and orientation
        new_state.x += new_state.velocity_x * dt
        new_state.y += new_state.velocity_y * dt
        new_state.heading += new_state.angular_velocity * dt
        
        # Normalize heading
        new_state.heading = (new_state.heading + math.pi) % (2 * math.pi) - math.pi
        
        # Update RPM based on speed and current gear
        if 1 <= new_state.gear <= len(self.setup.gear_ratios):
            gear_ratio = self.setup.gear_ratios[new_state.gear - 1] * self.setup.final_drive
            wheel_speed = speed / (self.track_width / 2)  # Simplified
            new_state.rpm = max(self.idle_rpm, wheel_speed * gear_ratio * 60 / (2 * math.pi))
        
        # Simple gear shifting logic
        if new_state.rpm > 12000 and new_state.gear < len(self.setup.gear_ratios):
            new_state.gear += 1
        elif new_state.rpm < 8000 and new_state.gear > 1:
            new_state.gear -= 1
        
        # Update tire temperature and wear (simplified)
        new_state.tire_temp += (abs(lateral_front + lateral_rear) / 1000 - 
                               (new_state.tire_temp - 80) * 0.01) * dt
        new_state.tire_wear += (speed / 100 + abs(front_slip + rear_slip)) * dt * self.setup.tire_wear_rate * 0.001
        
        return new_state
    
    def get_max_lateral_acceleration(self, speed: float) -> float:
        """
        Calculate maximum lateral acceleration at given speed
        
        Args:
            speed: Current speed (m/s)
            
        Returns:
            Maximum lateral acceleration (m/s²)
        """
        _, downforce_front, downforce_rear = self.calculate_aerodynamic_forces(speed)
        
        weight_front = self.mass * 9.81 * self.setup.weight_distribution
        weight_rear = self.mass * 9.81 * (1.0 - self.setup.weight_distribution)
        normal_front = weight_front + downforce_front
        normal_rear = weight_rear + downforce_rear
        
        max_force = self.tire_d * self.setup.tire_grip * (normal_front + normal_rear)
        max_accel = max_force / self.mass
        
        return max_accel
