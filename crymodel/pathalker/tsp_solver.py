# crymodel/pathalker/tsp_solver.py
"""TSP solver for pathwalking."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
import os

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


def solve_tsp_ortools(
    distance_matrix: np.ndarray,
    time_limit_seconds: int = 30,
    verbose: bool = True,
) -> tuple[list[int], float]:
    """Solve TSP using OR-Tools.
    
    Args:
        distance_matrix: (N, N) distance matrix in Å (integers recommended)
        time_limit_seconds: Time limit for solver
        verbose: Print solution information
        
    Returns:
        Tuple of (route as list of node indices, total path length in Å)
    """
    if not ORTOOLS_AVAILABLE:
        raise ImportError("OR-Tools not available. Install with: pip install ortools")
    
    n_nodes = len(distance_matrix)
    
    # Create data model
    data = {
        "distance_matrix": distance_matrix.tolist(),
        "num_vehicles": 1,
        "depot": 0,
    }
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(
        n_nodes, data["num_vehicles"], data["depot"]
    )
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        """Returns the distance between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        raise RuntimeError("TSP solver failed to find a solution")
    
    # Extract route
    route = []
    index = routing.Start(0)
    path_length_scaled = 0.0
    previous_node = None
    
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        if previous_node is not None:
            # Calculate distance using node indices (not routing indices)
            # Distance matrix was scaled by 100 and converted to integers
            path_length_scaled += distance_matrix[previous_node][node]
        previous_node = node
        index = solution.Value(routing.NextVar(index))
    
    # Final node (should be depot, index 0)
    final_node = manager.IndexToNode(index)
    if final_node != route[0]:  # Should return to start
        route.append(final_node)
        if previous_node is not None:
            path_length_scaled += distance_matrix[previous_node][final_node]
    
    # Divide by 100 to get back to Å
    path_length_angstrom = float(path_length_scaled) / 100.0
    
    if verbose:
        print(f"  TSP solution found: path length = {path_length_angstrom:.2f} Å")
        print(f"  Route: {route[:10]}{'...' if len(route) > 10 else ''}")
    
    return route, path_length_angstrom


def solve_tsp_lkh(
    distance_matrix: np.ndarray,
    lkh_executable: str = "LKH",
    working_dir: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[list[int], float]:
    """Solve TSP using LKH solver (requires LKH executable).
    
    Args:
        distance_matrix: (N, N) distance matrix in Å (integers recommended)
        lkh_executable: Path to LKH executable
        working_dir: Working directory for temporary files
        verbose: Print solution information
        
    Returns:
        Tuple of (route as list of node indices, total path length in Å)
    """
    if working_dir is None:
        working_dir = Path.cwd()
    else:
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
    
    # Write TSPLib file
    tsp_file = working_dir / "pathwalker.tsp"
    _write_tsplib_file(tsp_file, distance_matrix)
    
    # Write LKH parameter file
    par_file = working_dir / "pathwalker.par"
    tour_file = working_dir / "pathwalker.tour"
    _write_lkh_par_file(par_file, tsp_file, tour_file)
    
    # Run LKH
    if verbose:
        print(f"  Running LKH solver...")
    
    cmd = f"{lkh_executable} {par_file}"
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        raise RuntimeError(f"LKH solver failed with exit code {exit_code}")
    
    # Read tour file
    route = _read_lkh_tour_file(tour_file)
    
    # Calculate path length
    # Distance matrix was scaled by 100 and converted to integers
    path_length_scaled = 0.0
    for i in range(len(route) - 1):
        path_length_scaled += distance_matrix[route[i]][route[i + 1]]
    
    # Divide by 100 to get back to Å
    path_length_angstrom = float(path_length_scaled) / 100.0
    
    if verbose:
        print(f"  TSP solution found: path length = {path_length_angstrom:.2f} Å")
        print(f"  Route: {route[:10]}{'...' if len(route) > 10 else ''}")
    
    return route, path_length_angstrom


def _write_tsplib_file(filename: Path, distance_matrix: np.ndarray) -> None:
    """Write TSPLib format file."""
    n = len(distance_matrix)
    
    with open(filename, "w") as f:
        f.write(f"NAME: {filename.stem}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: Pathwalker TSP problem\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        
        for row in distance_matrix:
            f.write(" ".join([str(int(d)) for d in row]) + "\n")
        
        f.write("EOF\n")


def _write_lkh_par_file(par_file: Path, tsp_file: Path, tour_file: Path) -> None:
    """Write LKH parameter file."""
    with open(par_file, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
        f.write("PRECISION = 100\n")


def _read_lkh_tour_file(tour_file: Path) -> list[int]:
    """Read LKH tour file."""
    if not tour_file.exists():
        raise FileNotFoundError(f"Tour file not found: {tour_file}")
    
    route = []
    reading_tour = False
    
    with open(tour_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                reading_tour = True
                continue
            if reading_tour:
                if line == "-1" or line == "EOF":
                    break
                try:
                    node = int(line) - 1  # LKH uses 1-based indexing
                    if node >= 0:
                        route.append(node)
                except ValueError:
                    continue
    
    # Remove duplicate start/end node if present
    if len(route) > 1 and route[0] == route[-1]:
        route.pop()
    
    return route

