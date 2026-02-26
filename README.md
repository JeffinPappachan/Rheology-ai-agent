# Rheology AI Agent

An autonomous control system for rheological processes using machine learning and LangGraph for closed-loop control.

## Overview

This project implements an autonomous agent that controls rheological properties (specifically viscosity) in a fluid system using:

- **ANN-based Surrogate Model**: A neural network that predicts fluid viscosity and shear stress based on temperature and flow rate
- **LangGraph-based Control Agent**: A state machine that implements closed-loop control using proportional control logic
- **Autonomous Operation**: The system can autonomously adjust flow rates to maintain target viscosity within specified tolerances

## Project Structure

```
rheology-ai-agent/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── dataset.csv             # Training data
│   ├── raw/                    # Raw data storage
│   └── processed/              # Processed data storage
├── models/
│   ├── surrogate_model.pt      # Trained neural network model
│   └── scaler.pkl             # Data preprocessing scaler
├── notebooks/
│   └── experimentation.ipynb  # Jupyter notebook for experimentation
├── src/
│   ├── agent/                 # LangGraph agent implementation
│   │   ├── graph.py           # State machine definition
│   │   ├── nodes.py           # Agent node logic
│   │   └── state.py           # State definitions
│   ├── control/               # Control logic
│   │   └── stability.py       # Stability checking functions
│   ├── plant/                 # Simulated plant model
│   │   └── simulated_plant.py # Plant simulation
│   ├── surrogate/             # Machine learning components
│   │   ├── inference.py       # Model inference
│   │   ├── model.py           # Neural network architecture
│   │   └── train.py           # Model training
│   └── utils/                 # Utility functions
│       ├── logger.py          # Logging utilities
│       └── metrics.py         # Evaluation metrics
├── main.py                    # Entry point for autonomous operation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Key Components

### Surrogate Model (`src/surrogate/`)

- **model.py**: Defines the neural network architecture for predicting rheological properties
- **train.py**: Training script for the surrogate model using historical data
- **inference.py**: Real-time prediction interface for the trained model

### Control Agent (`src/agent/`)

- **graph.py**: Defines the LangGraph state machine with nodes for prediction, checking, decision-making, and control application
- **nodes.py**: Implements the logic for each agent node:
  - `predict_node`: Uses the surrogate model to predict viscosity
  - `check_node`: Compares predicted viscosity against target with tolerance
  - `decision_node`: Implements proportional control logic for shear-thinning fluids
  - `apply_node`: Applies control adjustments and manages iteration limits

### Control Logic (`src/control/`)

- **stability.py**: Simple stability checking to determine if the system is within acceptable bounds

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd rheology-ai-agent
```

2. Install dependencies:

```bash
uv install
```

## Usage

### Training the Surrogate Model

```bash
python src/surrogate/train.py
```

This will train the neural network model using the data in `data/dataset.csv` and save the trained model to `models/surrogate_model.pt`.

### Running the Autonomous Agent

```bash
python main.py
```

The main script will:

1. Display the feasible viscosity range at 75°C for various flow rates
2. Run the closed-loop control agent starting with initial conditions
3. Iteratively adjust the flow rate until the target viscosity is achieved or the maximum iterations are reached

### Configuration

Edit `config/config.yaml` to adjust control parameters:

```yaml
target_viscosity: 2.3 # Target viscosity value
tolerance: 0.1 # Acceptable deviation from target
max_iterations: 20 # Maximum control iterations
```

## Example Output

```
---- Feasible Viscosity Range at 75°C ----
Flow: 0.50  →  Viscosity: 3.4567
Flow: 1.00  →  Viscosity: 2.8901
Flow: 1.50  →  Viscosity: 2.3456
Flow: 2.00  →  Viscosity: 1.7890
Flow: 2.50  →  Viscosity: 1.2345
Flow: 3.00  →  Viscosity: 0.6789

---- Running Closed Loop Agent ----
Final State:
{
    "temperature": 75,
    "flow_rate": 1.52,
    "predicted_viscosity": 2.301,
    "deviation": false,
    "control_action": 0.02,
    "iteration": 3
}
```

## Technical Details

### Control Algorithm

The agent implements a proportional control strategy specifically designed for shear-thinning fluids:

- Uses the sign of the derivative relationship between viscosity and flow rate
- Applies proportional gain (Kp = 0.3) for stable convergence
- Includes safety bounds to prevent unrealistic flow rate values

### State Machine Architecture

The LangGraph implementation provides:

- Clear separation of concerns between prediction, evaluation, decision-making, and control
- Iterative control loops with convergence checking
- Built-in iteration limits to prevent infinite loops

### Machine Learning Integration

- Neural network surrogate model replaces complex physical simulations
- Real-time inference enables fast control decisions
- Data preprocessing ensures consistent input scaling

## Dependencies

- **Core ML**: PyTorch, scikit-learn, NumPy, Pandas
- **Control**: LangGraph for state machine implementation
- **Utilities**: PyYAML for configuration, joblib for model persistence
- **Visualization**: Matplotlib for training and analysis

## Contributing

1. Ensure all new features include appropriate tests
2. Update the configuration schema if adding new parameters
3. Document any changes to the control logic or model architecture
4. Follow the existing code style and structure

## License

MIT

## Contact

jeffinpappachan110@gmail.com
