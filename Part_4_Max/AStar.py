#TO-DO:
import heapq
import random
import math
import numpy as np
import timeit
import matplotlib.pyplot as plt
import time

"""
1. Implement a class for directed weighted graph
- We're gonna make it a adjacency matrix list to represent all the relationships, because it's easier to implement
- Steal from lab 3
2. Implement a class for A* search algorithm
- Uses priority queue to find lowest cost nodes
"""


class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def generate_random_graph(num_nodes, num_edges, max_edge_weight=100, min_edge_weight=1):
    #checking if the number of edges is valid
    if num_edges > num_nodes * (num_nodes - 1):
        raise ValueError("Too many edges for the given number of nodes.")

    graph = DirectedWeightedGraph() # initialize the graph
    for i in range(num_nodes):
        graph.add_node(i)

    edges_added = set() # tf does this do?

    while len(edges_added) < num_edges:

        node1, node2 = random.sample(range(num_nodes), 2)
        # Ensure that the edge is not already added and is not a self-loop
        if (node1, node2) not in edges_added:
            weight = random.randint(min_edge_weight, max_edge_weight)  # Assign a random weight between 1 and 100
            graph.add_edge(node1, node2, weight)
            edges_added.add((node1, node2))
        

    return graph







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
            path = reconstruct_path(predecessors, source, destination)
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

def reconstruct_path(predecessors, source, destination):
    #helper function: reconstructs path from source to destination.
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    # Check if the path starts with source; if not, no path was found.
    if path[0] == source:
        return path
    return []



def test_A_Star():

    for i in range(100):
        graph = generate_random_graph(300, 100, 1000, 1)
        heuristic = heuristic_gen(graph, 0, 4)
    # Create a random graph with 5 nodes and 10 edges
    graph = generate_random_graph(5, 10, 100, 1)
    
    # Generate a random heuristic for the graph
    heuristic = heuristic_gen(graph, 0, 4)
    
    # Run A* algorithm from node 0 to node 4
    predecessors, path = A_Star(graph, 0, 4, heuristic)
    
    print("Predecessors:", predecessors)
    print("Path from 0 to 4:", path)
    print("Graph nodes:", graph.adj.keys())

test_graph = generate_random_graph(5, 10, 100, 1)
print("Graph nodes:", test_graph.adj.keys())
print("Graph edges:", test_graph.weights.keys())