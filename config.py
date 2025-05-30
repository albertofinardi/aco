import os

# Determine the project's base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ACO Parameters
DEFAULT_CONFIG = {
    # Algorithm settings
    'algorithm_type': 'AS',  # 'AS' or 'MMAS'
    'use_local_search': False,
    
    # Core ACO parameters
    'alpha': 1.0,
    'beta': 2.5,
    'evaporation_rate': 0.5,
    'initial_pheromone': 1.0,
    
    # Execution settings
    'max_iterations': 100,
    'convergence_threshold': 1e-6,
    
    # Colony settings
    'colony_size': None,  # If None, will be set to number of cities
}

# Problem instances
TSP_INSTANCES = {
    'berlin52': {
        'file': os.path.join(DATA_DIR, 'berlin52.tsp'),
        'optimal': 7542
    },
    'small10': {
        'file': os.path.join(DATA_DIR, 'small10.tsp'),
        'optimal': None  # No known optimal for this test instance
    },
    'eil51': {
        'file': os.path.join(DATA_DIR, 'eil51.tsp'),
        'optimal': 426
    },
    'eil101': {
        'file': os.path.join(DATA_DIR, 'eil101.tsp'),
        'optimal': 629
    },
    'ch150': {
        'file': os.path.join(DATA_DIR, 'ch150.tsp'),
        'optimal': None
    },
}