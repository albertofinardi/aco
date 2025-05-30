import matplotlib.pyplot as plt
import time

class TourVisualizer:
    """Visualizer for real-time TSP solution updates."""
    
    def __init__(self, coordinates, title="ACO Algorithm Progress"):
        """
        Initialize the visualizer with the city coordinates.
        
        Args:
            coordinates: numpy array of city coordinates (x, y)
            title: Plot title
        """
        self.coordinates = coordinates
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.tour_line = None
        self.iteration_text = None
        self.length_text = None
        self.best_length = float('inf')
        self.best_tour = None
        self.start_time = time.time()
        
        self._setup_plot()
        
        plt.ion()
        plt.show(block=False)
    
    def _setup_plot(self):
        """Set up the initial plot elements."""
        self.ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                       c='red', s=50, zorder=10)
        
        for i, (x, y) in enumerate(self.coordinates):
            self.ax.text(x, y, str(i), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="circle", fc="white", ec="black", alpha=0.7),
                        zorder=15)
        
        self.tour_line, = self.ax.plot([], [], 'b-', linewidth=1.5, alpha=0.8, zorder=5)
        
        self.iteration_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                          verticalalignment='top', fontsize=10,
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        self.length_text = self.ax.text(0.02, 0.93, '', transform=self.ax.transAxes,
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        self.ax.set_title(self.title)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.margins(0.1)
        
        plt.tight_layout()
    
    def update(self, tour, tour_length, iteration, best_tour=None, best_length=None):
        """
        Update the visualization with a new tour.
        
        Args:
            tour: Current tour (list of city indices)
            tour_length: Length of the current tour
            iteration: Current iteration number
            best_tour: Best tour found so far (optional)
            best_length: Length of the best tour (optional)
        """
        if best_tour is not None and best_length is not None:
            self.best_tour = best_tour
            self.best_length = best_length
        else:
            if tour_length < self.best_length:
                self.best_tour = tour.copy()
                self.best_length = tour_length
        
        x_tour = [self.coordinates[city, 0] for city in tour]
        y_tour = [self.coordinates[city, 1] for city in tour]
        
        x_tour.append(self.coordinates[tour[0], 0])
        y_tour.append(self.coordinates[tour[0], 1])
        
        self.tour_line.set_data(x_tour, y_tour)
        
        current_time = time.time() - self.start_time
        self.iteration_text.set_text(f'Iteration: {iteration} | Time: {current_time:.1f}s')
        self.length_text.set_text(f'Current: {tour_length:.1f} | Best: {self.best_length:.1f}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.01)
    
    def save(self, filename):
        """Save the current plot to a file."""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    def close(self):
        """Close the plot."""
        plt.close(self.fig)
