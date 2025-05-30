import argparse
import time
import os
import multiprocessing
from config import DEFAULT_CONFIG, TSP_INSTANCES
from models.graph import Graph
from algo.aco import ACO
from utils.visualization import plot_tsp_solution, plot_convergence, plot_pheromone_heatmap
from utils.tsp_parser import load_tsp_data

def main():
    parser = argparse.ArgumentParser(description='Ant Colony Optimization for TSP')

    # General options
    parser.add_argument('--instance', type=str, default='berlin52', 
                        choices=list(TSP_INSTANCES.keys()),
                        help='TSP instance name')
    parser.add_argument('--algorithm', type=str, default=DEFAULT_CONFIG['algorithm_type'],
                        choices=['AS', 'MMAS'], help='ACO algorithm variant')
    parser.add_argument('--alpha', type=float, default=DEFAULT_CONFIG['alpha'],
                        help='Pheromone importance')
    parser.add_argument('--beta', type=float, default=DEFAULT_CONFIG['beta'],
                        help='Heuristic importance')
    parser.add_argument('--evaporation', type=float, default=DEFAULT_CONFIG['evaporation_rate'],
                        help='Pheromone evaporation rate')
    parser.add_argument('--iterations', type=int, default=DEFAULT_CONFIG['max_iterations'],
                        help='Maximum number of iterations')
    parser.add_argument('--local-search', action='store_true', 
                        default=DEFAULT_CONFIG['use_local_search'],
                        help='Use 2-opt local search')
    parser.add_argument('--colony-size', type=int, default=None,
                        help='Number of ants (default: number of cities)')
    
    # Visualization options
    parser.add_argument('--no-visualization', action='store_true', 
                        help='Disable all visualizations')
    parser.add_argument('--show-tour', action='store_true',
                        help='Show best tour visualization during simulation')
    parser.add_argument('--show-ants', action='store_true',
                        help='Show ant movement visualization during simulation')
    parser.add_argument('--show-pheromones', action='store_true',
                        help='Show pheromone levels in ant visualization')
    parser.add_argument('--show-pheromone-matrix', action='store_true',
                        help='Show pheromone matrix visualization during simulation')
    parser.add_argument('--viz-speed', type=float, default=1.0,
                        help='Visualization speed multiplier (default: 1.0)')
    parser.add_argument('--no-final-plots', action='store_true',
                        help='Disable final result plots')
    parser.add_argument('--plot-name', type=str, default=None,
                        help='Name for the final plots (default: None)')

    # Threading options
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                        help=f'Number of threads to use (default: {multiprocessing.cpu_count()} - all available cores)')
    parser.add_argument('--no-threading', action='store_true',
                        help='Disable multithreading (single-threaded execution)')
    
    args = parser.parse_args()
    
    if args.instance in TSP_INSTANCES:
        instance_data = TSP_INSTANCES[args.instance]
        tsp_file = instance_data['file']
        
        try:
            print(f"Loading {args.instance} from {tsp_file}")
            coordinates = load_tsp_data(tsp_file)
            
            optimal_length = instance_data['optimal']
            print(f"Loaded {args.instance} with {len(coordinates)} cities")
            if optimal_length:
                print(f"Known optimal tour length: {optimal_length}")
            else:
                print("Optimal tour length unknown for this instance")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please ensure the TSP file exists at: {tsp_file}")
            print("Hint: You may need to create the data files first")
            return
    else:
        print(f"Unknown instance: {args.instance}")
        return
    
    graph = Graph(coordinates=coordinates)
    
    num_threads = 1 if args.no_threading else args.threads
    
    show_tour = args.show_tour and not args.no_visualization
    show_ants = args.show_ants and not args.no_visualization
    show_pheromone_matrix = args.show_pheromone_matrix and not args.no_visualization
    
    aco = ACO(
        graph=graph,
        colony_size=args.colony_size,
        alpha=args.alpha,
        beta=args.beta,
        evaporation_rate=args.evaporation,
        algorithm_type=args.algorithm,
        use_local_search=args.local_search,
        visualization_speed=args.viz_speed,
        show_tour=show_tour,
        show_ants=show_ants,
        show_pheromone=show_pheromone_matrix,
        num_threads=num_threads
    )
    
    aco.show_pheromones_in_ant_viz = args.show_pheromones
    
    has_local_search = " + LS" if args.local_search else ""
    print(f"\nRunning {args.algorithm}{has_local_search} with alpha={args.alpha}, beta={args.beta}, evaporation={args.evaporation}")
    print(f"Visualization: {'None' if args.no_visualization else f'Tour={show_tour}, Ants={show_ants}, Pheromone Matrix={show_pheromone_matrix}'}")
    print(f"Using {num_threads} thread{'s' if num_threads > 1 else ''} for computation")
    
    start_time = time.time()
    best_tour, best_length = aco.run(max_iterations=args.iterations)
    end_time = time.time()
    
    print(f"\nResults:")
    print(f"Best tour length: {best_length}")
    if optimal_length:
        gap = (best_length - optimal_length) / optimal_length * 100
        print(f"Gap to optimal: {abs(gap):.2f}%")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    if not args.no_final_plots:
        plot_name = args.plot_name if args.plot_name else args.instance
        solution_plot = plot_tsp_solution(
            coordinates, best_tour, 
            title=f"{plot_name} - Best Tour Length: {best_length:.2f}"
        )
        solution_plot.savefig(os.path.join(results_dir, f"{plot_name}_solution.png"))
        
        convergence_plot = plot_convergence(
            aco.convergence_history,
            title=f"Convergence for {plot_name} using {args.algorithm} {has_local_search}"
        )
        convergence_plot.savefig(os.path.join(results_dir, f"{plot_name}_convergence.png"))
        
        pheromone_plot = plot_pheromone_heatmap(
            aco.pheromone_manager.pheromone_matrix,
            title=f"Final Pheromone Levels for {plot_name}"
        )
        pheromone_plot.savefig(os.path.join(results_dir, f"{plot_name}_pheromone.png"))
        
        print(f"Plots saved to '{results_dir}' directory")

if __name__ == "__main__":
    main()