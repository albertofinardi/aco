import numpy as np
from tqdm import tqdm
import time
import os
import platform
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ACO:
    """ACO algorithm with optimized threading model."""
    
    def __init__(self, graph, colony_size=None, alpha=1.0, beta=2.0, 
                 evaporation_rate=0.5, initial_pheromone=1.0, 
                 algorithm_type='AS', use_local_search=True,
                 visualization_speed=1.0, show_tour=False, show_ants=False,
                 show_pheromone=False, num_threads=None):
        """
        Initialize the ACO algorithm with visualization options and thread control.
        
        Args:
            graph: Graph instance representing the TSP
            colony_size: Number of ants in the colony
            alpha: Importance of pheromone
            beta: Importance of heuristic information
            evaporation_rate: Rate of pheromone evaporation
            initial_pheromone: Initial pheromone value
            algorithm_type: 'AS' (Ant System) or 'MMAS' (MAX-MIN Ant System)
            use_local_search: Whether to use 2-opt local search
            visualization_speed: Speed of visualization (higher = faster)
            show_tour: Whether to show the tour visualization
            show_ants: Whether to show the ant movement visualization
            show_pheromone: Whether to show the pheromone matrix visualization
            num_threads: Number of threads to use for parallel processing
        """
        from models.colony import Colony
        from algo.pheromone import PheromoneManager
        
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.use_local_search = use_local_search
        self.algorithm_type = algorithm_type
        self.visualization_speed = visualization_speed
        self.show_tour = show_tour
        self.show_ants = show_ants
        self.show_pheromone = show_pheromone
        
        self.num_threads = num_threads
        if num_threads is None or num_threads <= 0:
            self.num_threads = 1
        elif num_threads == 1:
            print("Running in single-threaded mode")
        else:
            max_threads = multiprocessing.cpu_count()
            self.num_threads = min(num_threads, max_threads)
            print(f"Running with {self.num_threads} threads")
        
        if graph.num_cities < 20 and self.num_threads > 1:
            print(f"Small problem size ({graph.num_cities} cities) detected. Forcing single-threaded operation for better performance.")
            self.num_threads = 1
        
        if self.num_threads > 1:
            self.batch_size = max(1, (graph.num_cities * graph.num_cities) // (self.num_threads * 1000))
            print(f"ACO using batch size of {self.batch_size} for parallel operations")
        else:
            self.batch_size = 1
        
        self.use_processes = False
        if self.num_threads > 1 and graph.num_cities > 100:
            self.use_processes = True
            print("Large problem detected. Using process-based parallelism for better performance.")
        
        colony_size = colony_size if colony_size is not None else graph.num_cities
        self.colony = Colony(graph, colony_size, self.num_threads)
        
        self.pheromone_manager = PheromoneManager(
            graph.num_cities, initial_pheromone, evaporation_rate
        )
        
        self.best_tour = None
        self.best_tour_length = float('inf')
        self.convergence_history = []
        
        self.viz_manager = None
        self.show_pheromones_in_ant_viz = True
        
        self.construction_time = 0
        self.local_search_time = 0
        self.pheromone_update_time = 0
    
    def run(self, max_iterations=100, convergence_threshold=None):
        """
        Run the ACO algorithm with live visualization.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if improvement is less than this threshold
            
        Returns:
            Best tour found and its length
        """
        from algo.local_search import two_opt, parallel_batch_two_opt
        
        try:
            from utils.visualization_manager import VisualizationManager
        except ImportError:
            print("Warning: VisualizationManager not found. Visualizations will be disabled.")
            self.show_tour = False
            self.show_ants = False
            self.show_pheromone = False
        
        if self.graph.coordinates is not None and (self.show_tour or self.show_ants or self.show_pheromone):
            try:
                self.viz_manager = VisualizationManager(self.graph.coordinates)
                
                if self.show_tour:
                    self.viz_manager.enable_tour_visualization(
                        title=f"{self.algorithm_type} Tour Progress - Alpha:{self.alpha} Beta:{self.beta}"
                    )
                
                if self.show_ants:
                    self.viz_manager.enable_ant_visualization(
                        title=f"{self.algorithm_type} Ant Movements - Alpha:{self.alpha} Beta:{self.beta}"
                    )
                
                if self.show_pheromone:
                    print("Initializing pheromone matrix visualization...")
                    self.viz_manager.enable_pheromone_visualization(
                        self.graph.num_cities,
                        title=f"{self.algorithm_type} Pheromone Evolution - Alpha:{self.alpha} Beta:{self.beta}"
                    )
                    print("Pheromone visualization initialized successfully")
            except Exception as e:
                print(f"Error initializing visualizations: {e}")
                import traceback
                traceback.print_exc()
                self.viz_manager = None
        
        progress_bar = tqdm(range(max_iterations))
        
        if self.show_pheromone and self.viz_manager and hasattr(self.viz_manager, 'pheromone_visualizer') and self.viz_manager.pheromone_visualizer:
            self.viz_manager.update_pheromone(
                self.pheromone_manager.pheromone_matrix,
                0,  # Iteration 0
                None,  # No best tour yet
                float('inf'),  # No best length yet
                "Initializing..."
            )
        
        for iteration in progress_bar:
            iteration_start = time.time()
            
            self.colony.reset_ants()
            
            construction_start = time.time()
            
            if self.show_ants:
                all_complete = False
                construction_steps = 0
                
                while not all_complete:
                    all_complete = self.colony.step_all_ants(
                        self.pheromone_manager.pheromone_matrix, self.alpha, self.beta
                    )
                    
                    construction_steps += 1
                    if construction_steps % max(1, int(5 / self.visualization_speed)) == 0:
                        if self.viz_manager:
                            try:
                                ant_positions = self.colony.get_ant_positions()
                                ant_paths = self.colony.get_ant_paths()
                                
                                current_best_tour = self.best_tour if self.best_tour is not None else []
                                current_best_length = self.best_tour_length if self.best_tour is not None else float('inf')
                                
                                self.viz_manager.update_ants(
                                    current_best_tour,
                                    current_best_length,
                                    iteration,
                                    ant_positions,
                                    ant_paths,
                                    self.pheromone_manager.pheromone_matrix if self.show_pheromones_in_ant_viz else None
                                )
                            except Exception as e:
                                print(f"Error updating ant visualization: {e}")
                
                tours = [ant.tour for ant in self.colony.ants]
                tour_lengths = [self.graph.total_distance(tour) for tour in tours]
            else:
                if self.num_threads > 1:
                    tours, tour_lengths = self.colony.construct_solutions(
                        self.pheromone_manager.pheromone_matrix, self.alpha, self.beta
                    )
                else:
                    tours = []
                    tour_lengths = []

                    for ant in self.colony.ants:
                        tour, tour_length = ant.construct_solution(
                            self.pheromone_manager.pheromone_matrix, self.alpha, self.beta
                        )
                        tours.append(tour)
                        tour_lengths.append(tour_length)
            
            self.construction_time += time.time() - construction_start
            
            if self.use_local_search:
                local_search_start = time.time()
                
                if self.num_threads > 1:
                    if self.use_processes:
                        # Process-based parallelism for larger problems
                        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                            chunk_size = max(1, len(tours) // self.num_threads)
                            futures = []
                            
                            for i in range(0, len(tours), chunk_size):
                                chunk_tours = tours[i:i+chunk_size]
                                future = executor.submit(
                                    parallel_batch_two_opt, 
                                    chunk_tours, 
                                    self.graph, 
                                    100,
                                    1
                                )
                                futures.append((future, i, i+chunk_size))
                            
                            for future, start_idx, end_idx in futures:
                                chunk_improved_tours, chunk_improved_lengths = future.result()
                                tours[start_idx:end_idx] = chunk_improved_tours
                                tour_lengths[start_idx:end_idx] = chunk_improved_lengths
                    else:
                        improved_tours, improved_lengths = parallel_batch_two_opt(
                            tours, self.graph, num_threads=self.num_threads
                        )
                        tours = improved_tours
                        tour_lengths = improved_lengths
                else:
                    for i, tour in enumerate(tours):
                        improved_tour, improved_length = two_opt(tour, self.graph)
                        tours[i] = improved_tour
                        tour_lengths[i] = improved_length
                
                self.local_search_time += time.time() - local_search_start
            
            # Find iteration best
            iteration_best_idx = np.argmin(tour_lengths)
            iteration_best_tour = tours[iteration_best_idx]
            iteration_best_length = tour_lengths[iteration_best_idx]
            
            # Update best solution
            if iteration_best_length < self.best_tour_length:
                self.best_tour = iteration_best_tour.copy()
                self.best_tour_length = iteration_best_length
            
            # Update pheromones based on algorithm type
            pheromone_start = time.time()
            if self.algorithm_type == 'AS':
                self.pheromone_manager.update_all_ants(tours, tour_lengths)
            elif self.algorithm_type == 'MMAS':
                # Only the best ant deposits pheromone
                self.pheromone_manager.update_best_ant(
                    iteration_best_tour, iteration_best_length
                )
            self.pheromone_update_time += time.time() - pheromone_start
            
            self.convergence_history.append(self.best_tour_length)
            
            progress_bar.set_description(
                f"Best: {self.best_tour_length:.2f}, Iteration best: {iteration_best_length:.2f}"
            )
            
            if self.viz_manager:
                try:
                    if self.show_tour:
                        self.viz_manager.update_tour(
                            iteration_best_tour,
                            iteration_best_length,
                            iteration,
                            self.best_tour,
                            self.best_tour_length
                        )
                    
                    if self.show_ants:
                        self.viz_manager.update_ants(
                            self.best_tour,
                            self.best_tour_length,
                            iteration,
                            self.colony.get_ant_positions(),
                            self.colony.get_ant_paths(),
                            self.pheromone_manager.pheromone_matrix if self.show_pheromones_in_ant_viz else None
                        )
                        
                    if self.show_pheromone:
                        pheromone_copy = self.pheromone_manager.pheromone_matrix.copy()
                        
                        min_pheromone = np.min(pheromone_copy[pheromone_copy > 0])
                        max_pheromone = np.max(pheromone_copy)
                        mean_pheromone = np.mean(pheromone_copy[pheromone_copy > 0])
                        
                        extra_info = f"Pheromone range: {min_pheromone:.4f} - {max_pheromone:.4f} (mean: {mean_pheromone:.4f})"
                        
                        self.viz_manager.update_pheromone(
                            pheromone_copy,
                            iteration,
                            self.best_tour,
                            self.best_tour_length,
                            extra_info
                        )
                except Exception as e:
                    print(f"Error updating visualizations: {e}")
            
            if convergence_threshold and iteration > 1:
                if abs(self.convergence_history[-2] - self.convergence_history[-1]) < convergence_threshold:
                    print(f"Converged after {iteration+1} iterations")
                    break
            
            iteration_time = time.time() - iteration_start
            target_time = 0.5 / self.visualization_speed
            if iteration_time < target_time:
                time.sleep(target_time - iteration_time)
        
        total_time = self.construction_time + self.local_search_time + self.pheromone_update_time
        print(f"\nPerformance statistics:")
        print(f"  Solution construction: {self.construction_time:.2f}s ({self.construction_time/total_time*100:.1f}%)")
        print(f"  Local search: {self.local_search_time:.2f}s ({self.local_search_time/total_time*100:.1f}%)")
        print(f"  Pheromone updates: {self.pheromone_update_time:.2f}s ({self.pheromone_update_time/total_time*100:.1f}%)")
        print(f"  Total algorithm time: {total_time:.2f}s")
        print(f"  Using {self.num_threads} thread{'s' if self.num_threads > 1 else ''} for parallel processing")
        
        if self.viz_manager:
            try:
                self.viz_manager.save_visualizations(f"{self.algorithm_type}_a{self.alpha}_b{self.beta}")
                self.viz_manager.close_all()
            except Exception as e:
                print(f"Error saving visualizations: {e}")
        
        return self.best_tour, self.best_tour_length