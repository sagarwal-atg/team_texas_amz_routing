import json
import folium
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

with open('/Users/Somi/Desktop/Projects/data/model_build_inputs/route_data.json') as f:
    route_data = json.load(f)

with open('/Users/Somi/Desktop/Projects/data/model_build_inputs/actual_sequences.json') as f:
    data_actual = json.load(f)
    
with open('/Users/Somi/Desktop/Projects/data/model_build_inputs/travel_times.json') as f:
    data_travel_time = json.load(f)

route_ids = list(route_data.keys())

route_number = 400

temp = {k: v for k, v in sorted(data_actual[route_ids[route_number]]['actual'].items(), key=lambda item: item[1])}
coordinates = []
for i in temp:
    lat = route_data[route_ids[route_number]]['stops'][i]['lat']
    long = route_data[route_ids[route_number]]['stops'][i]['lng']
    coordinates.append([lat, long])

matrix_order = list(temp.keys())

distances = []
for i in matrix_order:
    temp_d = []
    for j in matrix_order:
        temp_d.append(data_travel_time[route_ids[route_number]][i][j])
    distances.append(temp_d)

"""Simple travelling salesman problem between cities."""

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distances
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, solution, should_print=False):
    
    """Prints solution on console."""
    if should_print:
        print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    
    solution_collector = []
    
    while not routing.IsEnd(index):
        temp = manager.IndexToNode(index)
        solution_collector.append(temp)
        plan_output += ' {} ->'.format(temp)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    if should_print:
        print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    
    return solution_collector

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


# Print solution on console.
if solution:
    solution_collector = print_solution(manager, routing, solution)


temp_tsp = []

for i in solution_collector:
    temp_tsp.append(list(temp.keys())[list(temp.values()).index(i)])


# Actual sequence
temp = {k: v for k, v in sorted(data_actual[route_ids[route_number]]['actual'].items(), key=lambda item: item[1])}
coordinates = []
for i in temp:
    lat = route_data[route_ids[route_number]]['stops'][i]['lat']
    long = route_data[route_ids[route_number]]['stops'][i]['lng']
    coordinates.append([lat, long])

# tsp co-ordinates    
coordinates_tsp = []
for i in temp_tsp:
    lat = route_data[route_ids[route_number]]['stops'][i]['lat']
    long = route_data[route_ids[route_number]]['stops'][i]['lng']
    coordinates_tsp.append([lat, long])

# Create the map and add the line
m = folium.Map(location=[coordinates[5][0], coordinates[5][1]], zoom_start=13)
my_PolyLine=folium.PolyLine(locations=coordinates, weight=3)
my_PolyLine1=folium.PolyLine(locations=coordinates_tsp, weight=3, color="red")


m.add_child(my_PolyLine)
m.add_child(my_PolyLine1)
m.save("index.html")