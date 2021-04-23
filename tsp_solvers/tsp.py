from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from IPython import embed


def compute_tsp_solution(distance_matrix, stops_dict, depot=0):
    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = distance_matrix
        data['num_vehicles'] = 1
        data['depot'] = depot
        return data


    def get_tsp_solution(manager, routing, solution, should_print=False):

        """Prints solution on console."""
        index = routing.Start(0)
        route_distance = 0

        solution_collector = []

        while not routing.IsEnd(index):
            temp = manager.IndexToNode(index)
            solution_collector.append(temp)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        return solution_collector, route_distance

    """Entry point of the program."""
    # Instantiate the data problem.
    tsp_data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(tsp_data['distance_matrix']),
                                           tsp_data['num_vehicles'], tsp_data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return tsp_data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    solution_collector = get_tsp_solution(manager, routing, solution)

    tsp = []
    for i in solution_collector[0]:
        tsp.append(list(stops_dict.keys())[i])
    
    return tsp
