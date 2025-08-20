# AI-Powered Racing Car Setup and Lap Time Optimization

A comprehensive reinforcement learning system that combines car setup optimization with driving control to achieve optimal lap times in racing simulations. The system uses both PPO and SAC algorithms and validates performance against real Formula 1 data using FastF1.

## 🏎️ Features

- **Joint Optimization**: Simultaneously optimizes car setup parameters AND driving strategy
- **Multiple RL Algorithms**: PPO vs SAC comparison for racing applications
- **Realistic Physics**: 2D racing simulator with F1-inspired car physics including:
  - Pacejka tire model ("Magic Formula")
  - Aerodynamic forces (drag and downforce)
  - Bicycle car model with realistic dynamics
  - Gear shifting and engine modeling
- **Safety System**: Control Barrier Functions and safety shields prevent dangerous actions
- **Real Data Validation**: Compare AI performance with actual F1 telemetry using FastF1
- **Track Variety**: Multiple track types (oval, Monaco-inspired, figure-8)
- **Curriculum Learning**: Gradually increase difficulty during training

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for faster training)
- Windows/Linux/macOS

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/aryasadawrate19/ai-racing-optimization.git
cd ai-racing-optimization
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run a quick demo:**
```bash
python main.py --mode demo
```

## 🎮 Usage

### Training Modes

#### 1. Train a Single Agent
```bash
# Train PPO agent
python main.py --mode train --agent PPO --track oval --timesteps 50000

# Train SAC agent with setup optimization
python main.py --mode train --agent SAC --track monaco --timesteps 50000
```

#### 2. Compare PPO vs SAC
```bash
python main.py --mode compare --track oval --timesteps 30000
```

#### 3. Validate Against F1 Data
```bash
python main.py --mode validate --agent PPO --track monaco
```

#### 4. Safety System Demo
```bash
python main.py --mode safety --track oval
```

#### 5. Live Visualization
```bash
# Watch AI learn to drive in real-time
python watch_ai_drive.py

# Options:
# 1. Demo Mode (no installation required)
# 2. Live PPO Training
# 3. Live SAC Training
# 4. Render Trained Agent
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Execution mode: train, compare, validate, safety, demo | demo |
| `--agent` | RL agent: PPO, SAC, both | PPO |
| `--track` | Track: oval, monaco, figure8 | oval |
| `--timesteps` | Training timesteps | 30000 |
| `--no-setup` | Disable setup optimization | False |
| `--no-safety` | Disable safety system | False |
| `--render` | Enable visual rendering | False |

## 🏁 Project Structure

```
RL_F1/
├── src/
│   ├── simulator/          # Racing physics and environment
│   │   ├── car_physics.py     # Car dynamics and physics
│   │   ├── track.py           # Track representation
│   │   └── racing_env.py      # Gymnasium environment
│   ├── rl_agents/          # Reinforcement learning agents
│   │   └── racing_agents.py   # PPO and SAC implementations
│   ├── safety/             # Safety systems
│   │   └── safety_shield.py   # Control barrier functions
│   └── data_analysis/      # F1 data analysis
│       └── f1_data_analyzer.py # FastF1 integration
├── models/                 # Saved trained models
├── logs/                   # Training logs and tensorboard
├── plots/                  # Performance visualizations
├── data/                   # Track data and cache
├── notebooks/              # Jupyter notebooks for analysis
├── main.py                 # Main training script
├── watch_ai_drive.py       # Live visualization system
├── requirements.txt        # Dependencies
├── VISUALIZATION_GUIDE.md  # Detailed visualization guide
└── README.md              # This file
```

## 🔬 Technical Details

### Car Physics Model

The simulator uses a bicycle model with realistic F1-inspired parameters:

- **Mass**: 798 kg (F1 minimum weight)
- **Wheelbase**: 3.7 m
- **Engine**: 1000 HP, 15000 RPM redline
- **Aerodynamics**: Speed-dependent drag and downforce
- **Tires**: Pacejka Magic Formula with temperature and wear modeling

### Setup Parameters

The AI optimizes these car setup parameters:

1. **Aerodynamics**: Front/rear downforce balance, drag coefficient
2. **Mechanical**: Weight distribution, brake balance
3. **Tires**: Grip level, wear rate
4. **Gearing**: Gear ratios, final drive

### Safety System

Control Barrier Functions ensure safe operation by constraining:

- Lateral acceleration limits
- Track boundary violations
- Maximum speed limits
- Slip angle constraints
- Yaw rate limits

### Performance Metrics

The system tracks:

- **Lap Times**: Best, average, consistency
- **Safety**: Crash rate, off-track incidents
- **Efficiency**: Setup parameter optimization
- **Comparison**: Delta to real F1 performance

## 📊 Results and Analysis

### Expected Performance

Based on the methodology, the AI should achieve:

- **Lap Times**: Within 5-10% of real F1 times on similar tracks
- **Consistency**: Standard deviation < 1 second for lap times
- **Safety**: Crash rate < 5% with safety system enabled
- **Setup Optimization**: Automatic tuning of aerodynamic balance

### Validation Against F1 Data

The system uses FastF1 to compare against real telemetry:

- Speed profiles around corners
- Braking and acceleration patterns
- Sector time comparisons
- Setup parameter analysis

## 🛠️ Development and Research

### Novel Contributions

1. **Joint Optimization**: First system to combine setup AND control optimization
2. **PPO vs SAC Comparison**: Comprehensive evaluation for racing applications
3. **Safety Integration**: Practical safety barriers for high-speed racing
4. **Real Data Validation**: Sim-to-real transfer using public F1 data

### Future Enhancements

- **3D Physics**: Upgrade to full 3D simulation
- **Weather Conditions**: Rain, wind, temperature effects
- **Multi-Agent**: Racing against other AI cars
- **Advanced Tracks**: Real circuit laser-scan data
- **Pit Strategy**: Tire changes and fuel management

## 📈 Monitoring and Logging

### TensorBoard Integration

Monitor training progress:

```bash
tensorboard --logdir=logs
```

### Performance Plots

The system automatically generates:

- Lap time distributions
- Setup parameter evolution
- Safety intervention statistics
- Comparison charts

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Errors**: Ensure GPU drivers are updated
2. **Memory Issues**: Reduce batch size or use CPU training
3. **FastF1 Errors**: Check internet connection for data download
4. **Rendering Issues**: Install pygame and check display drivers

### Performance Tips

- Use GPU for training (10x speed improvement)
- Start with smaller timesteps for testing
- Enable tensorboard for monitoring
- Use curriculum learning for better convergence

## 📚 References

This project builds upon research in:

- Formula RL (Remonda et al., 2022)
- Safe RL for high-speed racing
- F1TENTH autonomous racing
- Pacejka tire modeling
- Control barrier functions

## 🤝 Contributing

Feel free to contribute by:

- Adding new track layouts
- Implementing additional RL algorithms
- Improving physics accuracy
- Adding more F1 data sources

## 📄 License

This project is for educational and research purposes.
Check [LICENSE.md](LICENSE.md) for more info

## 🏆 Acknowledgments

- FastF1 team for F1 data access
- Stable-Baselines3 for RL implementations
- F1TENTH community for racing inspiration
- OpenAI Gymnasium for environment framework

---

**Ready to race? Start your engines and let the AI find the perfect setup!** 🏎️💨
