import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class TrackPoint:
    """Single point on the racing track"""
    x: float                    # X coordinate (m)
    y: float                    # Y coordinate (m)
    heading: float             # Track heading at this point (radians)
    curvature: float           # Track curvature (1/radius, positive = left turn)
    elevation: float           # Track elevation (m)
    track_width: float         # Track width at this point (m)
    grip_level: float          # Grip level multiplier (0.5-1.5)
    distance: float            # Distance from start line (m)

class RaceTrack:
    """
    Racing track representation with centerline, boundaries, and track properties
    """
    
    def __init__(self, name: str = "Generic Track"):
        self.name = name
        self.centerline: List[TrackPoint] = []
        self.track_length: float = 0.0
        self.sector_boundaries: List[float] = []  # Sector end distances
        self.pit_lane_entry: Optional[float] = None
        self.pit_lane_exit: Optional[float] = None
        
    def create_oval_track(self, length: float = 2000.0, width: float = 200.0, 
                         track_width: float = 15.0) -> None:
        """
        Create a simple oval track for testing
        
        Args:
            length: Straight section length (m)
            width: Track width (distance between straights) (m)
            track_width: Racing surface width (m)
        """
        points = []
        total_distance = 0.0
        
        # Number of points for smooth curves
        straight_points = 50
        curve_points = 50
        
        # Straight section 1 (bottom)
        for i in range(straight_points):
            x = (i / (straight_points - 1)) * length
            y = 0.0
            points.append((x, y, 0.0, 0.0))  # heading=0, curvature=0
        
        # Turn 1 (right turn)
        radius = width / 2
        for i in range(curve_points):
            angle = (i / (curve_points - 1)) * math.pi
            x = length + radius * math.sin(angle)
            y = radius * (1 - math.cos(angle))
            heading = angle
            curvature = 1.0 / radius
            points.append((x, y, heading, curvature))
        
        # Straight section 2 (top)
        for i in range(straight_points):
            x = length - (i / (straight_points - 1)) * length
            y = width
            points.append((x, y, math.pi, 0.0))
        
        # Turn 2 (left turn)
        for i in range(curve_points):
            angle = math.pi + (i / (curve_points - 1)) * math.pi
            x = -radius * math.sin(angle - math.pi)
            y = width - radius * (1 + math.cos(angle - math.pi))
            heading = angle
            curvature = 1.0 / radius
            points.append((x, y, heading, curvature))
        
        # Convert to TrackPoint objects
        self.centerline = []
        for i, (x, y, heading, curvature) in enumerate(points):
            if i > 0:
                dx = x - points[i-1][0]
                dy = y - points[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            track_point = TrackPoint(
                x=x, y=y, heading=heading, curvature=curvature,
                elevation=0.0, track_width=track_width, grip_level=1.0,
                distance=total_distance
            )
            self.centerline.append(track_point)
        
        self.track_length = total_distance
        
        # Set sector boundaries (3 equal sectors)
        self.sector_boundaries = [
            self.track_length / 3,
            2 * self.track_length / 3,
            self.track_length
        ]
    
    def create_monaco_inspired_track(self) -> None:
        """
        Create a Monaco-inspired street circuit with various corner types
        """
        # This is a simplified version - in practice you'd load from track data
        points = []
        
        # Start/finish straight
        for i in range(30):
            x = i * 10.0
            y = 0.0
            points.append((x, y, 0.0, 0.0))
        
        # Hairpin turn (tight)
        center_x, center_y = 300.0, 50.0
        radius = 30.0
        for i in range(40):
            angle = (i / 39) * math.pi
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            heading = angle + math.pi/2
            curvature = 1.0 / radius
            points.append((x, y, heading, curvature))
        
        # Add more complex sections...
        # (For brevity, I'll keep this simple, but you can expand)
        
        # Convert to TrackPoint objects
        total_distance = 0.0
        self.centerline = []
        for i, (x, y, heading, curvature) in enumerate(points):
            if i > 0:
                dx = x - points[i-1][0]
                dy = y - points[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            # Vary grip levels for different track sections
            grip_level = 1.0
            if 100 < total_distance < 200:  # Slippery section
                grip_level = 0.8
            elif 400 < total_distance < 500:  # High grip section
                grip_level = 1.2
            
            track_point = TrackPoint(
                x=x, y=y, heading=heading, curvature=abs(curvature),
                elevation=0.0, track_width=12.0, grip_level=grip_level,
                distance=total_distance
            )
            self.centerline.append(track_point)
        
        self.track_length = total_distance
        self.sector_boundaries = [
            self.track_length / 3,
            2 * self.track_length / 3,
            self.track_length
        ]
    
    def get_track_info_at_distance(self, distance: float) -> TrackPoint:
        """
        Get track information at a specific distance along the centerline
        
        Args:
            distance: Distance along track (m)
            
        Returns:
            TrackPoint with interpolated values
        """
        # Wrap distance to track length
        distance = distance % self.track_length
        
        # Find the two nearest points
        if not self.centerline:
            raise ValueError("Track has no centerline points")
        
        # Binary search for efficiency with large tracks
        left, right = 0, len(self.centerline) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.centerline[mid].distance <= distance:
                left = mid
            else:
                right = mid
        
        p1 = self.centerline[left]
        p2 = self.centerline[right] if right < len(self.centerline) else self.centerline[0]
        
        # Handle wrap-around at track end
        if right == 0:
            p2_distance = p2.distance + self.track_length
        else:
            p2_distance = p2.distance
        
        # Interpolation factor
        if p2_distance == p1.distance:
            t = 0.0
        else:
            t = (distance - p1.distance) / (p2_distance - p1.distance)
        
        # Interpolate values
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)
        heading = self._interpolate_angle(p1.heading, p2.heading, t)
        curvature = p1.curvature + t * (p2.curvature - p1.curvature)
        elevation = p1.elevation + t * (p2.elevation - p1.elevation)
        track_width = p1.track_width + t * (p2.track_width - p1.track_width)
        grip_level = p1.grip_level + t * (p2.grip_level - p1.grip_level)
        
        return TrackPoint(
            x=x, y=y, heading=heading, curvature=curvature,
            elevation=elevation, track_width=track_width,
            grip_level=grip_level, distance=distance
        )
    
    def _interpolate_angle(self, angle1: float, angle2: float, t: float) -> float:
        """Interpolate between two angles, handling wraparound"""
        diff = angle2 - angle1
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        return angle1 + t * diff
    
    def get_closest_point_on_track(self, x: float, y: float) -> Tuple[float, float]:
        """
        Find the closest point on track centerline to given coordinates
        
        Args:
            x, y: World coordinates
            
        Returns:
            Tuple of (distance_along_track, lateral_offset)
        """
        min_distance = float('inf')
        closest_distance = 0.0
        lateral_offset = 0.0
        
        for point in self.centerline:
            dx = x - point.x
            dy = y - point.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                closest_distance = point.distance
                
                # Calculate lateral offset (positive = right of centerline)
                track_heading = point.heading
                to_car_heading = math.atan2(dy, dx)
                relative_heading = to_car_heading - track_heading
                lateral_offset = distance * math.sin(relative_heading)
        
        return closest_distance, lateral_offset
    
    def is_on_track(self, x: float, y: float, safety_margin: float = 0.5) -> bool:
        """
        Check if a position is on the racing surface
        
        Args:
            x, y: World coordinates
            safety_margin: Additional margin beyond track boundaries (m)
            
        Returns:
            True if position is on track
        """
        distance, lateral_offset = self.get_closest_point_on_track(x, y)
        track_info = self.get_track_info_at_distance(distance)
        
        max_offset = track_info.track_width / 2 + safety_margin
        return abs(lateral_offset) <= max_offset
    
    def get_sector_times(self, current_distance: float, 
                        sector_times: List[float]) -> List[Optional[float]]:
        """
        Calculate sector times based on current position
        
        Args:
            current_distance: Current distance along track
            sector_times: List to store sector times
            
        Returns:
            Updated sector times list
        """
        current_sector = 0
        for i, boundary in enumerate(self.sector_boundaries):
            if current_distance <= boundary:
                current_sector = i
                break
        
        return sector_times
    
    def save_track(self, filename: str) -> None:
        """Save track to JSON file"""
        track_data = {
            'name': self.name,
            'track_length': self.track_length,
            'sector_boundaries': self.sector_boundaries,
            'centerline': [
                {
                    'x': p.x, 'y': p.y, 'heading': p.heading,
                    'curvature': p.curvature, 'elevation': p.elevation,
                    'track_width': p.track_width, 'grip_level': p.grip_level,
                    'distance': p.distance
                }
                for p in self.centerline
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2)
    
    def load_track(self, filename: str) -> None:
        """Load track from JSON file"""
        with open(filename, 'r') as f:
            track_data = json.load(f)
        
        self.name = track_data['name']
        self.track_length = track_data['track_length']
        self.sector_boundaries = track_data['sector_boundaries']
        
        self.centerline = []
        for p_data in track_data['centerline']:
            point = TrackPoint(
                x=p_data['x'], y=p_data['y'], heading=p_data['heading'],
                curvature=p_data['curvature'], elevation=p_data['elevation'],
                track_width=p_data['track_width'], grip_level=p_data['grip_level'],
                distance=p_data['distance']
            )
            self.centerline.append(point)

class TrackGenerator:
    """Utility class for generating various track layouts"""
    
    @staticmethod
    def create_figure_eight(length: float = 1000.0) -> RaceTrack:
        """Create a figure-8 track"""
        track = RaceTrack("Figure Eight")
        # Implementation would go here
        track.create_oval_track(length, 200.0)  # Placeholder
        return track
    
    @staticmethod
    def create_spa_inspired() -> RaceTrack:
        """Create a Spa-Francorchamps inspired track with elevation changes"""
        track = RaceTrack("Spa Inspired")
        # Implementation would go here
        track.create_oval_track(2000.0, 300.0)  # Placeholder
        return track
    
    @staticmethod
    def create_silverstone_inspired() -> RaceTrack:
        """Create a Silverstone inspired track with high-speed corners"""
        track = RaceTrack("Silverstone Inspired")
        # Implementation would go here
        track.create_oval_track(1800.0, 250.0)  # Placeholder
        return track
