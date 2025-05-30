"""
Optimized Colony class with improved threading performance.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import multiprocessing
from .ant import Ant

class Colony:
    """A colony of ants for solving TSP with optimized threading."""
    
    def __init__(self, graph, colony_size=None, num_threads=None):
        """
        Initialize an ant colony.
        
        Args:
            graph: Graph instance representing the TSP
            colony_size: Number of ants in the colony (default: num_cities)
            num_threads: Number of threads to use (default: min(colony_size, CPU count))
        """
        self.graph = graph
        self.colony_size = colony_size if colony_size is not None else graph.num_cities
        
        # Set up threading
        if num_threads is None or num_threads <= 0:
            self.num_threads = 1  # Default to single thread if not specified
        else:
            max_threads = multiprocessing.cpu_count()
            self.num_threads = min(num_threads, max_threads)
        
        # Determine if parallelism is worth it based on problem size
        self.use_parallelism = self.num_threads > 1 and self.colony_size >= self.num_threads
        
        # If parallelism is worth it, determine batch size for optimal threading
        if self.use_parallelism:
            # Calculate the minimum work per thread to make parallelism beneficial
            # Rule of thumb: at least 10,000 operations per thread for TSP
            min_work_per_thread = 10000
            
            # Estimated work for a single ant = O(n²) where n is number of cities
            work_per_ant = self.graph.num_cities ** 2
            
            # Batch size = min work per thread / work per ant
            # Ensure at least one ant per batch
            self.batch_size = max(1, min_work_per_thread // work_per_ant)
            
            # Adjust batch size based on colony size and thread count
            # Each thread should handle multiple ants for efficiency
            ants_per_thread = max(1, self.colony_size // self.num_threads)
            self.batch_size = min(self.batch_size, ants_per_thread)
            
            # Log threading configuration
            print(f"Colony initialized with {self.colony_size} ants")
            print(f"Colony using {self.num_threads} threads with batch size {self.batch_size}")
        else:
            # If not using parallelism, optimize for sequential execution
            print(f"Colony initialized with {self.colony_size} ants (sequential execution)")
            self.batch_size = 1
        
        # Create ants with different starting positions
        self.reset_ants()
        
        self.best_tour = None
        self.best_tour_length = float('inf')
    
    def reset_ants(self):
        """Reset ants for a new iteration."""
        self.ants = []
        for i in range(self.colony_size):
            start_city = i % self.graph.num_cities  # Distribute ants evenly
            self.ants.append(Ant(self.graph, start_city))
    
    def _ant_construct_solution_batch(self, ant_indices, pheromone_matrix, alpha, beta):
        """
        Worker function for parallel solution construction of a batch of ants.
        
        Args:
            ant_indices: List of indices of ants to process
            pheromone_matrix: Matrix of pheromone values
            alpha: Exponent for pheromone importance
            beta: Exponent for heuristic importance
            
        Returns:
            List of (ant_idx, tour, tour_length) tuples
        """
        results = []
        for ant_idx in ant_indices:
            try:
                ant = self.ants[ant_idx]
                tour, tour_length = ant.construct_solution(pheromone_matrix, alpha, beta)
                results.append((ant_idx, tour, tour_length))
            except Exception as e:
                print(f"Error in ant construction for ant {ant_idx}: {e}")
                # Return a fallback result
                results.append((ant_idx, [], float('inf')))
        return results
    
    def construct_solutions(self, pheromone_matrix, alpha, beta):
        """
        Have all ants construct tours and track the best solution.
        Uses parallel or sequential execution based on configuration.
        
        Args:
            pheromone_matrix: Matrix of pheromone values
            alpha: Exponent for pheromone importance
            beta: Exponent for heuristic importance
            
        Returns:
            List of tours and their lengths
        """
        tours = [None] * self.colony_size
        tour_lengths = [None] * self.colony_size
        
        # Check if problem size is suitable for parallelism
        if self.use_parallelism:
            # Split ants into batches for parallel processing
            ant_batches = []
            for i in range(0, self.colony_size, self.batch_size):
                end = min(i + self.batch_size, self.colony_size)
                ant_batches.append(list(range(i, end)))
            
            # Run solution construction in parallel
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all batches
                future_to_batch = {}
                for batch in ant_batches:
                    future = executor.submit(
                        self._ant_construct_solution_batch,
                        batch, 
                        pheromone_matrix,
                        alpha,
                        beta
                    )
                    future_to_batch[future] = batch
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results = future.result()
                    for ant_idx, tour, tour_length in batch_results:
                        tours[ant_idx] = tour
                        tour_lengths[ant_idx] = tour_length
                        
                        # Update best tour if needed
                        if tour_length < self.best_tour_length:
                            self.best_tour = tour.copy()
                            self.best_tour_length = tour_length
        else:
            # Sequential execution for better performance on small problems
            for i, ant in enumerate(self.ants):
                tour, tour_length = ant.construct_solution(pheromone_matrix, alpha, beta)
                tours[i] = tour
                tour_lengths[i] = tour_length
                
                # Update best tour if needed
                if tour_length < self.best_tour_length:
                    self.best_tour = tour.copy()
                    self.best_tour_length = tour_length
        
        return tours, tour_lengths
    
    def step_all_ants(self, pheromone_matrix, alpha, beta):
        """
        Step all ants forward one city in their tour construction.
        This function is primarily used for visualization.
        
        Args:
            pheromone_matrix: Matrix of pheromone values
            alpha: Exponent for pheromone importance
            beta: Exponent for heuristic importance
            
        Returns:
            Boolean indicating if all ants have completed their tours
        """
        all_complete = True
        
        for ant in self.ants:
            # If the ant hasn't completed its tour yet
            if len(ant.tour) < self.graph.num_cities:
                all_complete = False
                
                # If the ant still has cities to visit
                if not np.all(ant.visited):
                    # Select the next city
                    next_city = ant.select_next_city(pheromone_matrix, alpha, beta)
                    
                    # Visit the next city
                    ant.tour.append(next_city)
                    ant.visited[next_city] = True
                    ant.current_city = next_city
                else:
                    # Calculate tour length once the tour is complete
                    ant.tour_length = self.graph.total_distance(ant.tour)
        
        return all_complete
    
    def get_ant_positions(self):
        """Get the current positions of all ants."""
        return [ant.current_city for ant in self.ants]
    
    def get_ant_paths(self):
        """Get the current partial paths of all ants."""
        return [ant.tour for ant in self.ants]