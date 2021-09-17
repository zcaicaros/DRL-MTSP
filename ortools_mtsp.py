"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(dist_mat):
    """Stores the data for the problem."""
    data = {'distance_matrix': dist_mat.tolist(), 'num_vehicles': 5, 'depot': 0}
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        # print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    # print('Maximum of the route distances: {}m'.format(max_route_distance))
    return max_route_distance


def solve(dist_mat, scale):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(dist_mat)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 1800

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    return print_solution(data, manager, routing, solution)/scale


def solve_multi_instances(instances, scale):
    objs = 0
    for i in range(instances.shape[0]):
        obj = solve(torch.cdist(instances[i] * scale, instances[i] * scale, p=2), scale)
        print('instance', i, ':', obj)
        objs += obj
    print('mean', objs/instances.shape[0])


if __name__ == '__main__':
    import torch

    n_nodes = 50
    batch_size = 1000
    scale = 10000
    # torch.manual_seed(1)
    # instances = torch.rand(size=[batch_size, n_nodes, 2])  # [batch, nodes, fea]

    instances = torch.load('./validation_data_'+str(n_nodes)+'_'+str(batch_size))  # [batch, nodes, fea]
    solve_multi_instances(instances, scale)
    # 10000 scale best is 2.10334
    # 1000 scale best is 2.10105
    # 100 scale best is 2.04879
