import numpy as np

def load_tsp_data(filename):
    """
    Load TSP data from a TSPLIB format file.
    
    Args:
        filename: Path to the TSP file
        
    Returns:
        numpy array of coordinates
    """
    coordinates = []
    reading_coords = False
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Start reading coordinates after NODE_COORD_SECTION
                if line == "NODE_COORD_SECTION":
                    reading_coords = True
                    continue
                
                # Stop reading at EOF
                if line == "EOF":
                    break
                    
                # Process coordinate lines
                if reading_coords:
                    try:
                        parts = line.split()
                        # TSPLIB format: index x y
                        if len(parts) >= 3:
                            node_id, x, y = parts[0], parts[1], parts[2]
                            coordinates.append((float(x), float(y)))
                    except ValueError:
                        print(f"Warning: Could not parse line: {line}")
                        continue
        
        if not coordinates:
            raise ValueError(f"No coordinates found in {filename}")
            
        return np.array(coordinates)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"TSP file not found: {filename}. Make sure the file exists in the specified path.")