"""Vehicles Routing Problem (VRP) with Time Windows."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def constrained_tsp(travel_time_matrix, time_window_list, depot):
    """Solve the VRP with time windows."""
    lamda_1 = 10000
    lamda_2 = 10000
    
    # Instantiate the data problem.
    data = {}
    data['time_matrix'] = travel_time_matrix
    data['time_windows'] = time_window_list
    data['num_vehicles'] = 1
    data['depot'] = depot

    def get_tsp_solution(manager, routing):

        """Prints solution on console."""
        solution_collector = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                temp = manager.IndexToNode(index)
                solution_collector.append(temp)
        return solution_collector

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        50000,  # allow waiting time
        50000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)

    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.SetCumulVarSoftLowerBound(index, int(time_window[0]), lamda_1)
        time_dimension.SetCumulVarSoftUpperBound(index, int(time_window[1]), lamda_2)
    
    
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    
    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    assert solution is not None

    return get_tsp_solution(manager, routing)
