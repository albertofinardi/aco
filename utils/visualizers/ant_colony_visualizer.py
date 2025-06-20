import matplotlib.pyplot as plt
import numpy as np
import time

class AntColonyVisualizer:
    """A real-time visualizer for ant movements in the ACO algorithm."""
    
    def __init__(self, coordinates, title="Ant Colony Movements"):
        """
        Initialize the visualizer with the city coordinates.
        
        Args:
            coordinates: numpy array of city coordinates (x, y)
            title: Plot title
        """
        self.coordinates = coordinates
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.best_tour_line = None
        self.ant_lines = []
        self.ant_markers = []
        self.pheromone_lines = []
        self.iteration_text = None
        self.length_text = None
        self.start_time = time.time()
        
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        self._setup_plot()
        
        plt.ion()
        plt.show(block=False)
    
    def _setup_plot(self):
        """Set up the initial plot elements."""
        self.ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                        c='black', s=50, zorder=10)
        
        for i, (x, y) in enumerate(self.coordinates):
            self.ax.text(x, y, str(i), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="circle", fc="white", ec="black", alpha=0.7),
                        zorder=15)
        
        self.best_tour_line, = self.ax.plot([], [], 'k--', linewidth=1.0, alpha=0.5, zorder=5)
        
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
    
    def update(self, best_tour, best_tour_length, iteration, ant_positions, ant_paths, pheromone_matrix=None):
        """
        Update the visualization with ant movements and the best tour so far.
        
        Args:
            best_tour: Best tour found so far
            best_tour_length: Length of the best tour
            iteration: Current iteration number
            ant_positions: List of current ant positions (city indices)
            ant_paths: List of current ant partial paths
            pheromone_matrix: Pheromone matrix (optional)
        """
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1
        if best_tour:
            x_best = [self.coordinates[city, 0] for city in best_tour]
            y_best = [self.coordinates[city, 1] for city in best_tour]
            x_best.append(self.coordinates[best_tour[0], 0])
            y_best.append(self.coordinates[best_tour[0], 1])
            self.best_tour_line.set_data(x_best, y_best)
        
        for line in self.ant_lines:
            line.remove()
        self.ant_lines = []
        
        for marker in self.ant_markers:
            marker.remove()
        self.ant_markers = []
        
        max_ants = min(len(ant_positions), 5)  # Limit to 5 ants for visualization
        
        if not hasattr(self, 'legend_created') or not self.legend_created:
            legend_elements = []
            for i in range(max_ants):
                legend_elements.append(plt.Line2D([0], [0], color=self.colors[i % 10], marker='o',
                                               markersize=6, markeredgecolor='black', linestyle='-',
                                               label=f'Ant {i+1}'))
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                          framealpha=0.7, title="Ants")
            self.legend_created = True
        
        for i in range(max_ants):
            if len(ant_paths[i]) > 1:
                x_path = [self.coordinates[city, 0] for city in ant_paths[i]]
                y_path = [self.coordinates[city, 1] for city in ant_paths[i]]
                
                line, = self.ax.plot(x_path, y_path, '-', color=self.colors[i % 10], 
                                   linewidth=1.5, alpha=0.7, zorder=6)
                self.ant_lines.append(line)
                
                if len(x_path) > 2:
                    for j in range(len(x_path)-1):
                        dx = x_path[j+1] - x_path[j]
                        dy = y_path[j+1] - y_path[j]
                        if dx*dx + dy*dy > (max(self.ax.get_xlim()) - min(self.ax.get_xlim()))**2 / 100:
                            self.ax.annotate('', 
                                            xy=(x_path[j+1], y_path[j+1]),
                                            xytext=(x_path[j], y_path[j]),
                                            arrowprops=dict(facecolor=self.colors[i % 10], 
                                                            edgecolor='none', 
                                                            width=2, headwidth=8, alpha=0.6),
                                            zorder=7)
                
                if ant_positions[i] is not None:
                    x_pos = self.coordinates[ant_positions[i], 0]
                    y_pos = self.coordinates[ant_positions[i], 1]
                    marker = self.ax.scatter(x_pos, y_pos, s=50, marker='o', 
                                         color=self.colors[i % 10], edgecolor='black', linewidth=1,
                                         alpha=0.8, zorder=20)
                    self.ant_markers.append(marker)
        
        if pheromone_matrix is not None:
            if not hasattr(self, 'pheromone_lines'):
                self.pheromone_lines = []
            else:
                for line in self.pheromone_lines:
                    line.remove()
                self.pheromone_lines = []
            
            max_pheromone = np.max(pheromone_matrix)
            min_pheromone = np.min(pheromone_matrix[pheromone_matrix > 0])
            
            percentile_threshold = 60  # Show top 40% of pheromone values
            threshold = np.percentile(pheromone_matrix[pheromone_matrix > 0], percentile_threshold)
            
            from matplotlib.colors import LinearSegmentedColormap
            pheromone_cmap = LinearSegmentedColormap.from_list(
                'pheromone', [(0, 'lightgreen'), (0.5, 'green'), (1, 'darkgreen')])
            
            if not hasattr(self, 'pheromone_legend_created') or not self.pheromone_legend_created:
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                
                sm = ScalarMappable(cmap=pheromone_cmap, norm=Normalize(vmin=min_pheromone, vmax=max_pheromone))
                sm.set_array([])
                cbar = self.fig.colorbar(sm, ax=self.ax, location='bottom', shrink=0.4, pad=0.1, 
                                       label='Pheromone Strength')
                self.pheromone_legend_created = True
            
            for i in range(len(self.coordinates)):
                for j in range(i+1, len(self.coordinates)):
                    pheromone_value = pheromone_matrix[i, j]
                    if pheromone_value > threshold:
                        normalized_value = (pheromone_value - min_pheromone) / (max_pheromone - min_pheromone)
                        normalized_value = min(1.0, max(0.0, normalized_value))
                        
                        width = 0.5 + 4.0 * normalized_value
                        alpha = 0.2 + 0.6 * normalized_value
                        
                        x = [self.coordinates[i, 0], self.coordinates[j, 0]]
                        y = [self.coordinates[i, 1], self.coordinates[j, 1]]
                        
                        color = pheromone_cmap(normalized_value)
                        line, = self.ax.plot(x, y, '-', color=color, linewidth=width, alpha=alpha, zorder=1)
                        self.pheromone_lines.append(line)
        
        current_time = time.time() - self.start_time
        
        complete_paths = sum(1 for path in ant_paths if len(path) == len(self.coordinates))
        
        self.iteration_text.set_text(f'Iteration: {iteration} | Step: {self.step_counter} | Time: {current_time:.1f}s')
        self.length_text.set_text(f'Best tour: {best_tour_length:.1f} | Complete paths: {complete_paths}/{len(ant_paths)}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        plt.pause(0.01)
    
    def save(self, filename):
        """Save the current plot to a file."""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    def close(self):
        """Close the plot."""
        plt.close(self.fig)