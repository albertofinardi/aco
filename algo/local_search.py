import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import multiprocessing

def two_opt_swap(tour, i, j):
    """
    Perform a 2-opt swap by reversing the segment between positions i and j.
    
    Args:
        tour: Current tour
        i, j: Positions to swap
        
    Returns:
        New tour after the swap
    """
    new_tour = tour.copy()
    new_tour[i:j+1] = tour[i:j+1][::-1]
    return new_tour

def two_opt(tour, graph, max_iterations=100):
    """
    Apply 2-opt local search to improve a tour.
    
    Args:
        tour: Initial tour
        graph: Graph instance with distance information
        max_iterations: Maximum number of improvement iterations
        
    Returns:
        Improved tour and its length
    """
    improved = True
    best_tour = tour.copy()
    best_length = graph.total_distance(tour)
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                # Skip adjacent edges
                if j - i == 1:
                    continue
                
                new_tour = two_opt_swap(best_tour, i, j)
                new_length = graph.total_distance(new_tour)
                
                # If the new tour is better, accept it
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break
            
            if improved:
                break
    
    return best_tour, best_length

def parallel_batch_two_opt(tours, graph, max_iterations=100, num_threads=None):
    """
    Apply 2-opt local search to multiple tours in parallel.
    
    Args:
        tours: List of tours to improve
        graph: Graph instance with distance information
        max_iterations: Maximum number of improvement iterations
        num_threads: Number of threads to use (default: min(len(tours), CPU count))
        
    Returns:
        List of improved tours and their lengths
    """
    max_threads = multiprocessing.cpu_count()
    
    if num_threads is None:
        num_threads = min(len(tours), max_threads)
    else:
        num_threads = min(num_threads, max_threads)
        num_threads = max(1, num_threads)
    
    improved_tours = [None] * len(tours)
    improved_lengths = [None] * len(tours)
    
    def process_tour(idx):
        """Process a single tour with 2-opt local search."""
        tour = tours[idx]
        improved_tour, improved_length = two_opt(tour, graph, max_iterations)
        return idx, improved_tour, improved_length
    
    # Run local search in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_tour = {
            executor.submit(process_tour, i): i for i in range(len(tours))
        }
        
        for future in concurrent.futures.as_completed(future_to_tour):
            try:
                idx, improved_tour, improved_length = future.result()
                improved_tours[idx] = improved_tour
                improved_lengths[idx] = improved_length
            except Exception as e:
                print(f"Error processing tour: {e}")
    
    return improved_tours, improved_lengths