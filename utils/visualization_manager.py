class VisualizationManager:
    """Manager class that provides access to different visualization types."""
    
    def __init__(self, coordinates):
        """
        Initialize the visualization manager.
        
        Args:
            coordinates: numpy array of city coordinates (x, y)
        """
        self.coordinates = coordinates
        self.tour_visualizer = None
        self.ant_visualizer = None
        self.pheromone_visualizer = None
    
    def enable_tour_visualization(self, title="ACO Algorithm Progress"):
        """
        Enable real-time visualization of the best tour.
        
        Args:
            title: Plot title
        
        Returns:
            TourVisualizer instance
        """
        from utils.visualizers.tour_visualizer import TourVisualizer
        self.tour_visualizer = TourVisualizer(self.coordinates, title)
        return self.tour_visualizer
    
    def enable_ant_visualization(self, title="Ant Colony Movements"):
        """
        Enable real-time visualization of ant movements.
        
        Args:
            title: Plot title
        
        Returns:
            AntColonyVisualizer instance
        """
        from utils.visualizers.ant_colony_visualizer import AntColonyVisualizer
        self.ant_visualizer = AntColonyVisualizer(self.coordinates, title)
        return self.ant_visualizer
    
    def enable_pheromone_visualization(self, num_cities, title="Pheromone Matrix Evolution"):
        """
        Enable real-time visualization of the pheromone matrix only (no tour).
        
        Args:
            num_cities: Number of cities in the TSP
            title: Plot title
        
        Returns:
            PheromoneVisualizer instance
        """
        from utils.visualizers.pheromone_visualizer import PheromoneVisualizer
        # Pass coordinates but the visualizer will not use them for display
        self.pheromone_visualizer = PheromoneVisualizer(num_cities, self.coordinates, title)
        return self.pheromone_visualizer
    
    def update_tour(self, tour, tour_length, iteration, best_tour=None, best_length=None):
        """
        Update the tour visualization if enabled.
        
        Args:
            tour: Current tour
            tour_length: Length of the current tour
            iteration: Current iteration number
            best_tour: Best tour found so far (optional)
            best_length: Length of the best tour (optional)
        """
        if self.tour_visualizer:
            self.tour_visualizer.update(tour, tour_length, iteration, best_tour, best_length)
    
    def update_ants(self, best_tour, best_tour_length, iteration, ant_positions, ant_paths, pheromone_matrix=None):
        """
        Update the ant visualization if enabled.
        
        Args:
            best_tour: Best tour found so far
            best_tour_length: Length of the best tour
            iteration: Current iteration number
            ant_positions: Current positions of ants
            ant_paths: Current partial paths of ants
            pheromone_matrix: Current pheromone matrix (optional)
        """
        if self.ant_visualizer:
            self.ant_visualizer.update(best_tour, best_tour_length, iteration, 
                                       ant_positions, ant_paths, pheromone_matrix)
    
    def update_pheromone(self, pheromone_matrix, iteration, best_tour=None, best_tour_length=None, extra_info=None):
        """
        Update the pheromone visualization if enabled.
        
        Args:
            pheromone_matrix: Current pheromone matrix
            iteration: Current iteration number
            best_tour: Best tour found so far (optional, not used for display)
            best_tour_length: Length of the best tour (optional)
            extra_info: Additional information to display (optional)
        """
        if self.pheromone_visualizer:
            self.pheromone_visualizer.update(pheromone_matrix, iteration, 
                                            best_tour, best_tour_length, extra_info)
    
    def save_visualizations(self, prefix="aco_solution"):
        """
        Save all active visualizations to files.
        
        Args:
            prefix: Prefix for the filenames
        """
        if self.tour_visualizer:
            self.tour_visualizer.save(f"{prefix}_tour.png")
        
        if self.ant_visualizer:
            self.ant_visualizer.save(f"{prefix}_ants.png")
            
        if self.pheromone_visualizer:
            self.pheromone_visualizer.save(f"{prefix}_pheromone.png")
    
    def close_all(self):
        """Close all visualizations."""
        if self.tour_visualizer:
            self.tour_visualizer.close()
        
        if self.ant_visualizer:
            self.ant_visualizer.close()
            
        if self.pheromone_visualizer:
            self.pheromone_visualizer.close()