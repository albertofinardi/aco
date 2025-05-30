import numpy as np

class Ant:
    """An ant agent that constructs TSP tours."""
    
    def __init__(self, graph, start_city=None):
        """
        Initialize an ant for tour construction.
        
        Args:
            graph: Graph instance representing the TSP
            start_city: Optional starting city, random if None
        """
        self.graph = graph
        self.num_cities = graph.num_cities
        
        # Initialize tour attributes
        self.tour = []
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.tour_length = 0
        
        # Set starting city
        self.current_city = start_city if start_city is not None else np.random.randint(0, self.num_cities)
        self.tour.append(self.current_city)
        self.visited[self.current_city] = True
    
    def select_next_city(self, pheromone_matrix, alpha, beta):
        """
        Select the next city to visit using the ACO probabilistic rule.
        
        Args:
            pheromone_matrix: Matrix of pheromone values
            alpha: Exponent for pheromone importance
            beta: Exponent for heuristic importance
            
        Returns:
            The index of the next city to visit
        """
        # Calculate probabilities for unvisited cities
        probabilities = np.zeros(self.num_cities)
        
        for city in range(self.num_cities):
            if not self.visited[city]:
                # Calculate pheromone and heuristic contributions
                pheromone = pheromone_matrix[self.current_city, city] ** alpha
                heuristic = self.graph.get_heuristic(self.current_city, city) ** beta
                probabilities[city] = pheromone * heuristic
        
        # Normalize probabilities
        total = np.sum(probabilities)
        if total > 0:
            probabilities = probabilities / total
        else:
            # If all pheromones are zero, use uniform distribution over unvisited cities
            for city in range(self.num_cities):
                if not self.visited[city]:
                    probabilities[city] = 1
            probabilities = probabilities / np.sum(probabilities)
        
        # Select next city using the probability distribution
        next_city = np.random.choice(self.num_cities, p=probabilities)
        return next_city
    
    def construct_solution(self, pheromone_matrix, alpha, beta):
        """
        Construct a complete TSP tour.
        
        Args:
            pheromone_matrix: Matrix of pheromone values
            alpha: Exponent for pheromone importance
            beta: Exponent for heuristic importance
            
        Returns:
            The constructed tour and its length
        """
        # Reset the ant for a new tour
        self.tour = [self.current_city]
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.visited[self.current_city] = True
        
        # Construct tour by visiting all unvisited cities
        while len(self.tour) < self.num_cities:
            next_city = self.select_next_city(pheromone_matrix, alpha, beta)
            self.tour.append(next_city)
            self.visited[next_city] = True
            self.current_city = next_city
        
        # Calculate tour length
        self.tour_length = self.graph.total_distance(self.tour)
        
        return self.tour, self.tour_length