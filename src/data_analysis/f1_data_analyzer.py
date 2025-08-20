import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache for faster data loading
fastf1.Cache.enable_cache('data/cache')

class F1DataAnalyzer:
    """
    Analyzer for F1 telemetry data using FastF1
    Used for validating AI performance against real racing data
    """
    
    def __init__(self, year: int = 2023):
        self.year = year
        self.sessions = {}
        self.telemetry_cache = {}
        
    def load_session(self, race: str, session_type: str = 'Q') -> fastf1.core.Session:
        """
        Load F1 session data
        
        Args:
            race: Race name (e.g., 'Monaco', 'Silverstone', 'Spa')
            session_type: Session type ('Q' for Qualifying, 'R' for Race, 'FP1', 'FP2', 'FP3')
            
        Returns:
            FastF1 session object
        """
        session_key = f"{race}_{session_type}"
        
        if session_key not in self.sessions:
            try:
                session = fastf1.get_session(self.year, race, session_type)
                session.load()
                self.sessions[session_key] = session
                print(f"Loaded {race} {session_type} session for {self.year}")
            except Exception as e:
                print(f"Error loading session {session_key}: {e}")
                return None
        
        return self.sessions[session_key]
    
    def get_fastest_lap_telemetry(self, race: str, driver: str = None, 
                                 session_type: str = 'Q') -> pd.DataFrame:
        """
        Get telemetry data for the fastest lap
        
        Args:
            race: Race name
            driver: Driver code (e.g., 'VER', 'HAM'). If None, gets overall fastest
            session_type: Session type
            
        Returns:
            DataFrame with telemetry data
        """
        session = self.load_session(race, session_type)
        if session is None:
            return pd.DataFrame()
        
        try:
            if driver:
                laps = session.laps.pick_driver(driver)
            else:
                laps = session.laps
            
            # Get fastest lap
            fastest_lap = laps.pick_fastest()
            
            if fastest_lap is None or fastest_lap.empty:
                print(f"No lap data found for {driver if driver else 'any driver'} in {race}")
                return pd.DataFrame()
            
            # Get telemetry
            telemetry = fastest_lap.get_telemetry()
            
            # Add calculated fields
            telemetry['LapTime'] = fastest_lap['LapTime']
            telemetry['Driver'] = fastest_lap['Driver']
            telemetry['Compound'] = fastest_lap['Compound']
            
            return telemetry
            
        except Exception as e:
            print(f"Error getting telemetry: {e}")
            return pd.DataFrame()
    
    def analyze_braking_zones(self, telemetry: pd.DataFrame) -> Dict:
        """
        Analyze braking zones from telemetry data
        
        Args:
            telemetry: Telemetry DataFrame
            
        Returns:
            Dictionary with braking analysis
        """
        if telemetry.empty:
            return {}
        
        # Identify braking zones (where brake > 0 and throttle == 0)
        braking_mask = (telemetry['Brake'] > 0) & (telemetry['Throttle'] == 0)
        braking_zones = []
        
        # Find continuous braking segments
        braking_starts = telemetry[braking_mask & ~braking_mask.shift(1).fillna(False)]
        braking_ends = telemetry[braking_mask & ~braking_mask.shift(-1).fillna(False)]
        
        for start_idx, end_idx in zip(braking_starts.index, braking_ends.index):
            zone = telemetry.loc[start_idx:end_idx]
            if len(zone) > 5:  # Minimum zone length
                braking_zones.append({
                    'start_distance': zone['Distance'].iloc[0],
                    'end_distance': zone['Distance'].iloc[-1],
                    'entry_speed': zone['Speed'].iloc[0],
                    'exit_speed': zone['Speed'].iloc[-1],
                    'max_brake_pressure': zone['Brake'].max(),
                    'duration': len(zone) / 10.0,  # Assuming 10Hz data
                    'distance': zone['Distance'].iloc[-1] - zone['Distance'].iloc[0]
                })
        
        analysis = {
            'num_braking_zones': len(braking_zones),
            'braking_zones': braking_zones,
            'total_braking_distance': sum(zone['distance'] for zone in braking_zones),
            'avg_brake_pressure': telemetry[telemetry['Brake'] > 0]['Brake'].mean(),
            'max_brake_pressure': telemetry['Brake'].max()
        }
        
        return analysis
    
    def analyze_cornering_performance(self, telemetry: pd.DataFrame) -> Dict:
        """
        Analyze cornering performance
        
        Args:
            telemetry: Telemetry DataFrame
            
        Returns:
            Dictionary with cornering analysis
        """
        if telemetry.empty:
            return {}
        
        # Calculate lateral acceleration (simplified)
        # In real F1 data, this would be more complex
        telemetry['LateralAccel'] = np.abs(np.gradient(telemetry['Speed']) * np.sin(np.radians(telemetry.get('SteeringAngle', 0))))
        
        # Identify corners (high lateral acceleration zones)
        corner_threshold = telemetry['LateralAccel'].quantile(0.7)
        corner_mask = telemetry['LateralAccel'] > corner_threshold
        
        corner_speeds = telemetry[corner_mask]['Speed']
        
        analysis = {
            'avg_corner_speed': corner_speeds.mean(),
            'min_corner_speed': corner_speeds.min(),
            'max_corner_speed': corner_speeds.max(),
            'avg_lateral_accel': telemetry['LateralAccel'].mean(),
            'max_lateral_accel': telemetry['LateralAccel'].max(),
            'corner_entry_speeds': [],
            'corner_exit_speeds': []
        }
        
        return analysis
    
    def get_sector_times(self, race: str, driver: str = None, 
                        session_type: str = 'Q') -> Dict:
        """
        Get sector times for analysis
        
        Args:
            race: Race name
            driver: Driver code
            session_type: Session type
            
        Returns:
            Dictionary with sector time analysis
        """
        session = self.load_session(race, session_type)
        if session is None:
            return {}
        
        try:
            if driver:
                laps = session.laps.pick_driver(driver)
            else:
                laps = session.laps
            
            fastest_lap = laps.pick_fastest()
            
            if fastest_lap is None or fastest_lap.empty:
                return {}
            
            sector_times = {
                'sector_1': fastest_lap['Sector1Time'].total_seconds(),
                'sector_2': fastest_lap['Sector2Time'].total_seconds(),
                'sector_3': fastest_lap['Sector3Time'].total_seconds(),
                'lap_time': fastest_lap['LapTime'].total_seconds()
            }
            
            return sector_times
            
        except Exception as e:
            print(f"Error getting sector times: {e}")
            return {}
    
    def compare_with_ai_performance(self, ai_telemetry: Dict, 
                                   race: str, driver: str = None) -> Dict:
        """
        Compare AI performance with real F1 data
        
        Args:
            ai_telemetry: AI telemetry data
            race: Race name for F1 data
            driver: F1 driver for comparison
            
        Returns:
            Comparison analysis
        """
        # Get F1 reference data
        f1_telemetry = self.get_fastest_lap_telemetry(race, driver)
        
        if f1_telemetry.empty:
            return {"error": "No F1 data available for comparison"}
        
        # Analyze both datasets
        f1_braking = self.analyze_braking_zones(f1_telemetry)
        f1_cornering = self.analyze_cornering_performance(f1_telemetry)
        f1_sectors = self.get_sector_times(race, driver)
        
        # Compare lap times
        f1_lap_time = f1_sectors.get('lap_time', 0)
        ai_lap_time = ai_telemetry.get('lap_time', 0)
        
        comparison = {
            'lap_time_comparison': {
                'f1_time': f1_lap_time,
                'ai_time': ai_lap_time,
                'delta': ai_lap_time - f1_lap_time,
                'delta_percentage': ((ai_lap_time - f1_lap_time) / f1_lap_time * 100) if f1_lap_time > 0 else 0
            },
            'speed_comparison': {
                'f1_max_speed': f1_telemetry['Speed'].max(),
                'f1_avg_speed': f1_telemetry['Speed'].mean(),
                'ai_max_speed': ai_telemetry.get('max_speed', 0),
                'ai_avg_speed': ai_telemetry.get('avg_speed', 0)
            },
            'braking_comparison': {
                'f1_braking_zones': f1_braking.get('num_braking_zones', 0),
                'f1_total_braking_distance': f1_braking.get('total_braking_distance', 0),
                'ai_braking_efficiency': ai_telemetry.get('braking_efficiency', 0)
            },
            'sector_comparison': {
                'f1_sectors': f1_sectors,
                'ai_sectors': ai_telemetry.get('sector_times', {})
            }
        }
        
        return comparison
    
    def visualize_telemetry(self, telemetry: pd.DataFrame, save_path: str = None):
        """
        Create telemetry visualization plots
        
        Args:
            telemetry: Telemetry DataFrame
            save_path: Path to save the plot
        """
        if telemetry.empty:
            print("No telemetry data to visualize")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Speed
        axes[0].plot(telemetry['Distance'], telemetry['Speed'])
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].set_title('Speed Trace')
        axes[0].grid(True)
        
        # Throttle and Brake
        axes[1].plot(telemetry['Distance'], telemetry['Throttle'], label='Throttle', color='green')
        axes[1].plot(telemetry['Distance'], telemetry['Brake'], label='Brake', color='red')
        axes[1].set_ylabel('Input (%)')
        axes[1].set_title('Throttle and Brake Input')
        axes[1].legend()
        axes[1].grid(True)
        
        # Gear
        if 'nGear' in telemetry.columns:
            axes[2].plot(telemetry['Distance'], telemetry['nGear'])
            axes[2].set_ylabel('Gear')
            axes[2].set_title('Gear Selection')
            axes[2].grid(True)
        
        # RPM
        if 'RPM' in telemetry.columns:
            axes[3].plot(telemetry['Distance'], telemetry['RPM'])
            axes[3].set_ylabel('RPM')
            axes[3].set_xlabel('Distance (m)')
            axes[3].set_title('Engine RPM')
            axes[3].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_performance_report(self, race: str, driver: str = None) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            race: Race name
            driver: Driver code
            
        Returns:
            Performance report dictionary
        """
        telemetry = self.get_fastest_lap_telemetry(race, driver)
        
        if telemetry.empty:
            return {"error": "No telemetry data available"}
        
        braking_analysis = self.analyze_braking_zones(telemetry)
        cornering_analysis = self.analyze_cornering_performance(telemetry)
        sector_times = self.get_sector_times(race, driver)
        
        report = {
            'race': race,
            'driver': driver,
            'year': self.year,
            'lap_time': sector_times.get('lap_time', 0),
            'sector_times': sector_times,
            'speed_stats': {
                'max_speed': telemetry['Speed'].max(),
                'avg_speed': telemetry['Speed'].mean(),
                'min_speed': telemetry['Speed'].min(),
                'speed_std': telemetry['Speed'].std()
            },
            'braking_performance': braking_analysis,
            'cornering_performance': cornering_analysis,
            'throttle_stats': {
                'avg_throttle': telemetry['Throttle'].mean(),
                'max_throttle': telemetry['Throttle'].max(),
                'throttle_application_time': (telemetry['Throttle'] > 50).sum() / len(telemetry)
            }
        }
        
        return report

class F1TrackGenerator:
    """
    Generate track layouts based on real F1 circuits
    """
    
    def __init__(self):
        # Approximate track data for major F1 circuits
        self.track_data = {
            'Monaco': {
                'length': 3337,
                'corners': 19,
                'elevation_change': 42,
                'avg_speed': 161,
                'characteristics': ['tight', 'street_circuit', 'elevation']
            },
            'Silverstone': {
                'length': 5891,
                'corners': 18,
                'elevation_change': 20,
                'avg_speed': 233,
                'characteristics': ['high_speed', 'flowing', 'medium_downforce']
            },
            'Spa': {
                'length': 7004,
                'corners': 19,
                'elevation_change': 101,
                'avg_speed': 234,
                'characteristics': ['high_speed', 'elevation', 'low_downforce']
            },
            'Monza': {
                'length': 5793,
                'corners': 11,
                'elevation_change': 23,
                'avg_speed': 264,
                'characteristics': ['very_high_speed', 'low_downforce', 'long_straights']
            }
        }
    
    def get_track_characteristics(self, track_name: str) -> Dict:
        """Get characteristics of a specific F1 track"""
        return self.track_data.get(track_name, {})
    
    def generate_reference_lap_time(self, track_name: str, car_performance: float = 1.0) -> float:
        """
        Generate a reference lap time based on track characteristics
        
        Args:
            track_name: Name of the F1 track
            car_performance: Performance multiplier (1.0 = current F1 performance)
            
        Returns:
            Estimated lap time in seconds
        """
        track_info = self.track_data.get(track_name, {})
        
        if not track_info:
            return 90.0  # Default lap time
        
        # Base lap time calculation (simplified)
        length = track_info['length']
        avg_speed = track_info['avg_speed'] * car_performance
        
        # Convert average speed from km/h to m/s
        avg_speed_ms = avg_speed / 3.6
        
        # Base time
        base_time = length / avg_speed_ms
        
        # Adjust for track characteristics
        if 'tight' in track_info.get('characteristics', []):
            base_time *= 1.2
        elif 'high_speed' in track_info.get('characteristics', []):
            base_time *= 0.9
        
        return base_time

def create_ai_vs_f1_comparison(ai_results: Dict, f1_analyzer: F1DataAnalyzer, 
                              track_name: str = "Monaco") -> Dict:
    """
    Create comprehensive comparison between AI and F1 performance
    
    Args:
        ai_results: AI performance results
        f1_analyzer: F1 data analyzer instance
        track_name: Track name for comparison
        
    Returns:
        Comprehensive comparison report
    """
    # Get F1 reference data
    f1_report = f1_analyzer.generate_performance_report(track_name)
    
    if "error" in f1_report:
        # Use estimated data if real F1 data not available
        track_gen = F1TrackGenerator()
        f1_estimated_time = track_gen.generate_reference_lap_time(track_name)
        
        comparison = {
            'track': track_name,
            'data_source': 'estimated',
            'f1_reference_time': f1_estimated_time,
            'ai_best_time': ai_results.get('best_lap_time', float('inf')),
            'performance_gap': ai_results.get('best_lap_time', float('inf')) - f1_estimated_time,
            'ai_completion_rate': 1.0 - ai_results.get('crash_rate', 1.0),
            'notes': 'Comparison based on estimated F1 performance due to data availability'
        }
    else:
        # Use real F1 data
        f1_lap_time = f1_report['lap_time']
        ai_lap_time = ai_results.get('best_lap_time', float('inf'))
        
        comparison = {
            'track': track_name,
            'data_source': 'real_f1_data',
            'f1_reference_time': f1_lap_time,
            'ai_best_time': ai_lap_time,
            'performance_gap': ai_lap_time - f1_lap_time,
            'performance_percentage': ((ai_lap_time - f1_lap_time) / f1_lap_time * 100) if f1_lap_time > 0 else 0,
            'ai_completion_rate': 1.0 - ai_results.get('crash_rate', 1.0),
            'f1_reference_data': f1_report,
            'ai_consistency': ai_results.get('consistency', float('inf'))
        }
    
    return comparison
