#TO-DO:
import heapq
import random
import math
import numpy as np
import timeit
import matplotlib.pyplot as plt
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import Helper.graphs as graphs


"""
1. Implement a class for directed weighted graph
- We're gonna make it a adjacency matrix list to represent all the relationships, because it's easier to implement
- Steal from lab 3
2. Implement a class for A* search algorithm
- Uses priority queue to find lowest cost nodes
"""


def heuristic_gen(G, source, destination, min, max): # for now we'll just use a random heuristic
    # but this can be improved in the future,
    heuristic = []
    for i in range(G.number_of_nodes()):
        heuristic.append(random.randint(min , max))
    return heuristic




    #Note: The heuristic "function" is just a dictionary of the nodes and their heuristic values
def a_star(graph, source, destination, heuristic):
    g_score = {node: float('inf') for node in graph.adj}
    g_score[source] = 0

    predecessor = {node: None for node in graph.adj}

    heap = []
    heapq.heappush(heap, (heuristic[source], source))

    closed = set()

    while heap:
        f_score, current = heapq.heappop(heap)

        if current == destination:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = predecessor[current]
            return predecessor , list(reversed(path))

        if current in closed:
            continue
        closed.add(current)

        for neighbor in graph.adjacent_nodes(current):
            tentative_g = g_score[current] + graph.w(current, neighbor)

            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                predecessor[neighbor] = current
                f_score = tentative_g + heuristic.get(neighbor, float('inf'))
                heapq.heappush(heap, (f_score, neighbor))

    return {} , []  # No path found


def test_A_Star():

    for i in range(100):
        graph = graphs.generate_random_graph(300, 100, 1000)
        heuristic = heuristic_gen(graph, 0, 4)
    # Create a random graph with 5 nodes and 10 edges
    graph = graphs.generate_random_graph(5, 10, 100)
    
    # Generate a random heuristic for the graph
    heuristic = heuristic_gen(graph, 0, 4)
    
    # Run A* algorithm from node 0 to node 4
    predecessors, path = A_Star(graph, 0, 4, heuristic)
    
    print("Predecessors:", predecessors)
    print("Path from 0 to 4:", path)
    print("Graph nodes:", graph.adj.keys())

test_graph = graphs.generate_random_graph(5, 10, 100)
print("Graph nodes:", test_graph.adj.keys())
print("Graph edges:", test_graph.weights.keys())