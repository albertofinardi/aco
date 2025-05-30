import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
from matplotlib.colors import LogNorm

class PheromoneVisualizer:
    """A real-time visualizer for the pheromone matrix only, with no tour display."""
    
    def __init__(self, num_cities, coordinates=None, title="Pheromone Matrix Evolution"):
        """
        Initialize the pheromone visualizer.
        
        Args:
            num_cities: Number of cities in the TSP
            coordinates: Optional numpy array of city coordinates (x, y) - not used for visualization
            title: Plot title
        """
        self.num_cities = num_cities
        self.title = title
        self.last_update_time = time.time()
        self.update_interval = 0.1
        
        print("Using matplotlib backend:", matplotlib.get_backend()) # MacOS related issue
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
            
        self.pheromone_image = None
        self.colorbar = None
        
        self.iteration_text = None
        self.info_text = None
        
        self.prev_pheromone = None
        self.prev_iteration = -1
        
        self._setup_plot()

        plt.ion()
        self.fig.canvas.draw()
        plt.show(block=False)
        
        print("Pheromone-only visualizer initialized")
    
    def _setup_plot(self):
        """Set up the initial plot elements."""
        print("Setting up pheromone visualization...")
        
        initial_data = np.ones((self.num_cities, self.num_cities))
        np.fill_diagonal(initial_data, 0)  # Zero on diagonal
        self.prev_pheromone = initial_data.copy()
        
        self.pheromone_image = self.ax.imshow(
            initial_data,
            cmap='viridis',
            norm=LogNorm(vmin=0.1, vmax=10),
            interpolation='nearest'
        )
        
        self.colorbar = self.fig.colorbar(self.pheromone_image, ax=self.ax, label='Pheromone Level')
        
        self.ax.set_xticks(np.arange(self.num_cities))
        self.ax.set_yticks(np.arange(self.num_cities))
        self.ax.set_xticklabels(np.arange(self.num_cities))
        self.ax.set_yticklabels(np.arange(self.num_cities))
        
        self.ax.set_xticks(np.arange(-.5, self.num_cities, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.num_cities, 1), minor=True)
        self.ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.2)
        
        self.ax.set_title("Pheromone Matrix")
        self.ax.set_xlabel('City j')
        self.ax.set_ylabel('City i')
        
        self.iteration_text = self.ax.text(0.02, 0.98, 'Iteration: 0', transform=self.ax.transAxes,
                                         verticalalignment='top', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        self.info_text = self.ax.text(0.02, 0.92, 'Initializing...', transform=self.ax.transAxes,
                                      verticalalignment='top', fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        self.fig.suptitle(self.title, fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        print("Pheromone visualization setup complete")
    
    def update(self, pheromone_matrix, iteration, best_tour=None, best_tour_length=None, extra_info=None):
        """
        Update the visualization with new pheromone matrix values.
        
        Args:
            pheromone_matrix: Current pheromone matrix
            iteration: Current iteration number
            best_tour: Best tour found so far (not used for visualization)
            best_tour_length: Length of the best tour
            extra_info: Additional information to display
        """
        current_time = time.time()
        if (iteration == self.prev_iteration and np.array_equal(pheromone_matrix, self.prev_pheromone)) or \
           (current_time - self.last_update_time < self.update_interval):
            return
            
        self.prev_pheromone = pheromone_matrix.copy()
        self.prev_iteration = iteration
        self.last_update_time = current_time
        
        try:
            matrix_to_display = pheromone_matrix.copy()
            
            min_positive = 1e-10
            matrix_to_display[matrix_to_display <= 0] = min_positive
            
            self.pheromone_image.set_data(matrix_to_display)
            
            valid_values = matrix_to_display[matrix_to_display > min_positive]
            if len(valid_values) > 0:
                vmin = max(min_positive, np.min(valid_values))
                vmax = max(0.1, np.max(valid_values))
                current_norm = self.pheromone_image.norm
                if not isinstance(current_norm, LogNorm) or \
                   (vmin < current_norm.vmin * 0.5 or vmin > current_norm.vmin * 2 or
                    vmax < current_norm.vmax * 0.5 or vmax > current_norm.vmax * 2):
                    self.pheromone_image.norm = LogNorm(vmin=vmin, vmax=vmax)
                    
                    self.colorbar.update_normal(self.pheromone_image)
            
            self.iteration_text.set_text(f'Iteration: {iteration}')
            
            if best_tour_length is not None:
                info_text = f'Best tour length: {best_tour_length:.2f}'
                if extra_info:
                    info_text += f'\n{extra_info}'
                self.info_text.set_text(info_text)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Error updating pheromone visualization: {e}")
    
    def close(self):
        """Close the plot."""
        try:
            plt.close(self.fig)
        except Exception as e:
            print(f"Error closing pheromone visualization: {e}")
    
    def save(self, filename):
        """Save the current plot to a file."""
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving pheromone visualization: {e}")