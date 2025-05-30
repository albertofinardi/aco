# Ant Colony Optimization for the Traveling Salesman Problem

## Overview

This project implements multiple Ant Colony Optimization variants inspired by the seminal work of Dorigo, Birattari, and Stützle (2006). The implementation focuses on the Traveling Salesman Problem as a benchmark for evaluating swarm intelligence algorithms.

### Implemented Algorithms

- **Ant System (AS)**: The original ACO algorithm with all ants contributing to pheromone updates
- **MAX-MIN Ant System (MMAS)**: Enhanced variant with pheromone bounds and elite-only updates
- **2-opt Local Search**: Optional tour improvement procedure
- **Real-time Visualization**: Live tracking of algorithm progress and ant movements

## Features

- Multiple ACO algorithm variants (AS, MMAS)
- 2-opt local search optimization
- Real-time visualization of algorithm progress
- Multi-threaded parallel execution
- Performance analysis
- Parameter configuration options
- Support for standard TSPLIB format files
- Detailed logging and convergence tracking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```bash
python main.py --instance small10 --iterations 50
```

## Quick Start

Run a basic simulation on the Berlin52 dataset:

```bash
python main.py --instance berlin52 --algorithm MMAS --local-search --show-tour
```

This will:
- Load the Berlin52 TSP instance (52 cities)
- Use MAX-MIN Ant System algorithm
- Apply 2-opt local search
- Display real-time tour visualization
- Run for 100 iterations (default)

## Running Simulations

### Basic Usage

```bash
python main.py [OPTIONS]
```

### Common Examples

**Basic run with visualization:**
```bash
python main.py --instance berlin52 --algorithm MMAS --show-tour --show-pheromone-matrix
```

**Performance comparison:**
```bash
python main.py --instance berlin52 --algorithm AS --iterations 200 --no-visualization
python main.py --instance berlin52 --algorithm MMAS --iterations 200 --no-visualization
```

**Parameter sensitivity analysis:**
```bash
python main.py --instance berlin52 --alpha 1.5 --beta 3.0 --evaporation 0.6 --show-tour
```

## Parameter Reference

### Core Algorithm Parameters

| Parameter | Flag | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Algorithm Type | `--algorithm` | `AS` | `AS`, `MMAS` | ACO algorithm variant |
| Alpha (α) | `--alpha` | `1.0` | `0.1-5.0` | Pheromone importance weight |
| Beta (β) | `--beta` | `2.5` | `0.1-10.0` | Heuristic information weight |
| Evaporation Rate (ρ) | `--evaporation` | `0.5` | `0.1-0.9` | Pheromone evaporation rate |
| Colony Size | `--colony-size` | `n_cities` | `1-500` | Number of ants in colony |
| Max Iterations | `--iterations` | `100` | `1-1000` | Maximum number of iterations |

### Algorithm Enhancement Options

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Local Search | `--local-search` | `False` | Enable 2-opt tour improvement |
| Multithreading | `--threads` | `CPU count` | Number of parallel threads |
| No Threading | `--no-threading` | `False` | Force single-threaded execution |

### Visualization Controls

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Show Tour | `--show-tour` | `False` | Real-time best tour visualization |
| Show Ants | `--show-ants` | `False` | Real-time ant movement visualization (only a part of the total colony size) |
| Show Pheromones | `--show-pheromones` | `False` | Display pheromone levels on the tour map |
| Show Pheronome Matrix | `--show-pheromone-matrix` | `False` | Real-time pheromone matrix |
| Visualization Speed | `--viz-speed` | `1.0` | Visualization update rate multiplier |
| No Visualization | `--no-visualization` | `False` | Disable all real-time visualization |
| No Final Plots | `--no-final-plots` | `False` | Skip generating result plots |
| Plot name | `--no-final-plots` | `Instance` | Name for the plots

In order to check real-time costruction of the solution `--show-tour` and `--show-pheromone-matrix` are suggested.

## Adding New TSP Data

### Supported Formats

The implementation supports TSPLIB format files. To add a new TSP instance:

### 1. Prepare the Data File

Create a `.tsp` file in TSPLIB format:

```
NAME: your_instance_name
TYPE: TSP
COMMENT: Description of your instance
DIMENSION: 25
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 60.0 200.0
2 180.0 200.0
3 80.0 180.0
...
25 100.0 100.0
EOF
```

### 2. Add to Data Directory

Place your `.tsp` file in the `data/` directory:

```
data/
├── berlin52.tsp
├── ...
└── your_instance.tsp  # Your new file
```

### 3. Register in Configuration

Edit `config.py` to add your instance:

```python
TSP_INSTANCES = {
    'berlin52': {
        'file': os.path.join(DATA_DIR, 'berlin52.tsp'),
        'optimal': 7542
    },
    ...
    # Add your instance here
}
```

### 4. Run Your Instance

```bash
python main.py --instance your_instance --show-tour
```

### Data Sources

- **GitHub**: https://github.com/mastqe/tsplib
- **Custom instances**: Generate using the format above

## Codebase Structure

```
├── main.py                     # Main entry point and CLI interface
├── config.py                   # Configuration and TSP instance definitions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── algo/                       # Core ACO algorithms
│   ├── aco.py                 # Main ACO orchestration class
│   ├── local_search.py        # 2-opt local search implementation
│   └── pheromone.py           # Pheromone management system
│
├── models/                     # Data structures and agent models
│   ├── ant.py                 # Individual ant agent implementation
│   ├── colony.py              # Ant colony management
│   └── graph.py               # TSP graph representation
│
├── utils/                          # Utilities and support functions
│   ├── tsp_parser.py               # TSPLIB format file parser
│   ├── visualization.py            # Static plotting functions
│   ├── visualization_manager.py    # Visualization coordination
│   └── visualizers/                # Real-time visualization components
│       ├── ant_colony_visualizer.py    # Ant movement visualization
│       ├── tour_visualizer.py          # Best tour visualization
│       └── visualization_manager.py    # Visualization coordination
│
├── data/                       # TSP instance files
│   ├── berlin52.tsp            # 52-city Berlin instance
│   ├── eil51.tsp               # 51-city Eilon instance  
│   ├── ch150.tsp               # 150-city Churritz instance
│   ├── eil101.tsp              # 101-city Eilon instance
│   └── small10.tsp             # 10-city test instance
│
└── results/                    # Generated outputs
```

### Core Components

#### Algorithm Layer (`algo/`)

**`aco.py`**: Main algorithm orchestration
- Coordinates the ACO metaheuristic
- Manages iteration loops and convergence
- Integrates visualization and local search
- Handles performance tracking

**`local_search.py`**: 2-opt improvement
- Implements 2-opt edge swap optimization
- Supports parallel tour improvement
- Provides significant quality enhancement

**`pheromone.py`**: Pheromone management
- Maintains pheromone matrix
- Handles evaporation and deposition
- Supports different ACO variants (AS, MMAS)

#### Model Layer (`models/`)

**`graph.py`**: TSP problem representation
- Stores city coordinates and distance matrix
- Provides efficient distance calculations
- Supports both coordinate and matrix input

**`ant.py`**: Individual ant behavior
- Implements probabilistic city selection
- Maintains tour construction state
- Supports step-by-step visualization

**`colony.py`**: Population management
- Coordinates multiple ants
- Handles parallel execution
- Manages ant initialization and reset

#### Utilities (`utils/`)

**`tsp_parser.py`**: File input handling
- Parses TSPLIB format files
- Extracts city coordinates
- Handles various TSP file formats

**`visualization.py`**: Static plotting
- Generates final result plots
- Creates convergence charts
- Produces pheromone heatmaps

**Visualizers**: Real-time display
- Live algorithm progress tracking
- Interactive ant movement display
- Dynamic pheromone visualization

## Visualization Features

### Real-time Visualization

#### Tour Visualization (`--show-tour`)
- Displays current best tour in real-time
- Shows iteration progress and solution quality
- Updates tour length and gap to optimal
- Color-coded cities and edges

#### Ant Movement Visualization (`--show-ants`)
- Shows individual ant paths during construction
- Displays pheromone trails (if enabled)
- Animates solution building process
- Multiple (not all) ant tracking with color coding

#### Pheromone Visualization (`--show-pheromones`)
- Real-time pheromone level display
- Color-coded edge intensities
- Shows learning process evolution
- Adaptive threshold visualization

### Static Result Plots

Generated automatically in `results/` directory:

- **Solution Plot**: Final best tour visualization
- **Convergence Plot**: Solution quality over iterations  
- **Pheromone Heatmap**: Final pheromone distribution matrix

## Examples

### Basic Examples

**Quick test run:**
```bash
python main.py --instance small10 --iterations 50 --show-tour
```

**Compare algorithms:**
```bash
# Ant System
python main.py --instance berlin52 --algorithm AS --iterations 100

# MAX-MIN Ant System  
python main.py --instance berlin52 --algorithm MMAS --iterations 100
```

### Advanced Examples

**Parameter sensitivity study:**
```bash
# Test different alpha values
python main.py --instance berlin52 --alpha 0.5 --show-tour
python main.py --instance berlin52 --alpha 1.0 --show-tour  
python main.py --instance berlin52 --alpha 2.0 --show-tour
```

**Full suggested visualization demo:**
```bash
python main.py --instance berlin52 --algorithm MMAS --show-tour \
    --show-pheromone-matrix --local-search
```

## Performance Tips

### Optimization Strategies

1. **Disable visualization** for performance runs:
   ```bash
   python main.py --instance berlin52 --no-visualization
   ```

2. **Use multithreading** for large instances:
   ```bash
   python main.py --instance berlin52 --threads 8
   ```
   By default, the code runs multithreaded on CPU count

3. **Enable local search** for better solutions:
   ```bash
   python main.py --instance berlin52 --local-search
   ```

4. **Adjust colony size** for problem complexity:
   ```bash
   python main.py --instance berlin52 --colony-size 100
   ```

### Memory Considerations

- Large instances (>100 cities) require significant memory for distance matrices
- Visualization uses additional memory for plot generation
- Consider `--no-visualization` for memory-constrained environments

## Acknowledgments

- Based on the foundational work of Dorigo, Birattari, and Stützle (2006)
- Inspired by the original Ant System algorithm by Marco Dorigo