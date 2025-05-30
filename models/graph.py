import numpy as np
from scipy.spatial.distance import pdist, squareform

class Graph:
    """Representation of a TSP problem as a complete graph."""
    
    def __init__(self, coordinates=None, distance_matrix=None):
        """
        Initialize a TSP graph either from coordinates or a distance matrix.
        
        Args:
            coordinates: List of (x, y) coordinates for each city
            distance_matrix: Pre-computed distance matrix
        """
        self.num_cities = None
        self.coordinates = None
        self.distances = None
        
        if coordinates is not None:
            self.coordinates = np.array(coordinates)
            self.num_cities = len(coordinates)
            # Compute distances between all pairs of points
            self.distances = squareform(pdist(self.coordinates, 'euclidean'))
        elif distance_matrix is not None:
            self.distances = np.array(distance_matrix)
            self.num_cities = len(distance_matrix)
        else:
            raise ValueError("Either coordinates or distance_matrix must be provided")
    
    def get_distance(self, city_i, city_j):
        """Get distance between two cities."""
        return self.distances[city_i, city_j]
    
    def get_heuristic(self, city_i, city_j):
        """
        Get heuristic value between two cities.
        For TSP, this is typically 1/distance.
        """
        distance = self.distances[city_i, city_j]
        # Avoid division by zero
        if distance == 0:
            return 1e10
        return 1.0 / distance
    
    def total_distance(self, tour):
        """Calculate the total distance of a tour."""
        total = 0
        for i in range(len(tour) - 1):
            total += self.distances[tour[i], tour[i+1]]
        # Add distance from last to first city to complete the loop
        total += self.distances[tour[-1], tour[0]]
        return total