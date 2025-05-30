import numpy as np

class PheromoneManager:
    """Manages pheromone levels and updates in the ACO algorithm."""
    
    def __init__(self, num_cities, initial_pheromone=1.0, evaporation_rate=0.5):
        """
        Initialize the pheromone manager.
        
        Args:
            num_cities: Number of cities in the TSP
            initial_pheromone: Initial pheromone level on all edges
            evaporation_rate: Rate of pheromone evaporation (0-1)
        """
        self.num_cities = num_cities
        self.evaporation_rate = evaporation_rate
        
        self.pheromone_matrix = np.ones((num_cities, num_cities)) * initial_pheromone
        
        np.fill_diagonal(self.pheromone_matrix, 0)
    
    def evaporate(self):
        """Apply pheromone evaporation to all edges."""
        self.pheromone_matrix *= (1 - self.evaporation_rate)
    
    def deposit_pheromone(self, tour, tour_length, deposit_amount=1.0):
        """
        Deposit pheromone on a tour based on its quality.
        
        Args:
            tour: List of cities in the order visited
            tour_length: Total length of the tour
            deposit_amount: Base amount to deposit
        """
        amount = deposit_amount / tour_length
        
        for i in range(len(tour) - 1):
            city_i, city_j = tour[i], tour[i+1]
            self.pheromone_matrix[city_i, city_j] += amount
            self.pheromone_matrix[city_j, city_i] += amount 
        
        city_i, city_j = tour[-1], tour[0]
        self.pheromone_matrix[city_i, city_j] += amount
        self.pheromone_matrix[city_j, city_i] += amount
    
    def update_all_ants(self, tours, tour_lengths):
        """
        Update pheromones based on all ants' tours (Ant System approach).
        
        Args:
            tours: List of tours constructed by ants
            tour_lengths: List of tour lengths
        """
        self.evaporate()
        
        for tour, tour_length in zip(tours, tour_lengths):
            self.deposit_pheromone(tour, tour_length)
    
    def update_best_ant(self, best_tour, best_tour_length):
        """
        Update pheromones based only on the best tour (MAX-MIN Ant System approach).
        
        Args:
            best_tour: The best tour found
            best_tour_length: Length of the best tour
        """
        self.evaporate()
        self.deposit_pheromone(best_tour, best_tour_length, deposit_amount=1.0)