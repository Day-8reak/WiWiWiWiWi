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
def A_Star(graph, source, destination, heuristic):
    openSet = []    # setting up priority queue

    heapq.heappush(openSet, (heuristic[source], source))    # Each element in the queue is a tuple: (f_score, current_node)
    
    # Dictionaries to track the best cost (g score) and predeccesor for every node
    gScore = {node: float('inf') for node in graph}
    gScore[source] = 0 # source node set to 0
    
    predecessors = {node: None for node in graph} # setting all predecessors to None
    
    # set to keep track of processed nodes
    closedSet = set()
    
    while openSet: # while there are nodes in the queue
        current = heapq.heappop(openSet)
        
        # If destination reached, reconstruct the path and return.
        if current == destination:
            path = graphs.reconstruct_path(predecessors, source, destination)
            return (predecessors, path)
        
        if current in closedSet:
            continue
        closedSet.add(current) # adding current node to closed set
        
        for n, w in graph[current].items(): # Check all the neighbors of current node
            currentGScore = gScore[current] + w
            
            if currentGScore < gScore[n]:
                # This path to neighbor is better than any previous one.
                gScore[n] = currentGScore
                predecessors[n] = current
                # f = g + heuristic
                fScore = currentGScore + heuristic.get(n, 0)
                heapq.heappush(openSet, (fScore, n))
                
    # If we get here, no path was found.
    return (predecessors, [])


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