import matplotlib.pyplot as plt

def plot_tsp_solution(coordinates, tour, title="TSP Solution"):
    """
    Plot a TSP solution.
    
    Args:
        coordinates: Array of (x, y) coordinates for each city
        tour: Sequence of city indices forming the tour
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=50)
    
    for i in range(len(tour) - 1):
        city_i, city_j = tour[i], tour[i+1]
        plt.plot(
            [coordinates[city_i, 0], coordinates[city_j, 0]],
            [coordinates[city_i, 1], coordinates[city_j, 1]],
            'b-'
        )
    
    city_i, city_j = tour[-1], tour[0]
    plt.plot(
        [coordinates[city_i, 0], coordinates[city_j, 0]],
        [coordinates[city_i, 1], coordinates[city_j, 1]],
        'b-'
    )

    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=12)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.tight_layout()
    
    return plt

def plot_convergence(convergence_history, title="ACO Convergence"):
    """
    Plot the convergence of the ACO algorithm.
    
    Args:
        convergence_history: List of best tour lengths at each iteration
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length')
    plt.grid(True)
    plt.tight_layout()
    
    return plt

def plot_pheromone_heatmap(pheromone_matrix, title="Pheromone Levels"):
    """
    Plot a heatmap of pheromone levels.
    
    Args:
        pheromone_matrix: Matrix of pheromone values
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(pheromone_matrix, cmap='viridis')
    plt.colorbar(label='Pheromone Level')
    plt.title(title)
    plt.xlabel('City j')
    plt.ylabel('City i')
    plt.tight_layout()
    
    return plt