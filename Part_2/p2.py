import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import tracemalloc
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import Helper.graphs as graphs

# Dijkstra's with relaxation limit k
def dijkstra(graph, source, k):
    distance = {}
    predecessor = {}
    relax_count = {}

    for node in graph.adj:
        distance[node] = float('inf')
        predecessor[node] = None
        relax_count[node] = 0

    distance[source] = 0

    heap = graphs.MinHeap()
    heap.push((0, source))

    while not heap.is_empty():
        dist_u, u = heap.pop()

        for neighbor in graph.adjacent_nodes(u):
            weight = graph.w(u, neighbor)
            new_dist = dist_u + weight

            if relax_count[neighbor] < k and new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                predecessor[neighbor] = u
                relax_count[neighbor] += 1
                heap.push((new_dist, neighbor))

    # Reconstruct paths from the predecessor dictionary
    path = {}
    for node in graph.adj:
        if distance[node] == float('inf'):
            path[node] = []  # No path
        else:
            # Reconstruct path from source to node
            cur = node
            rev_path = []
            while cur is not None:
                rev_path.append(cur)
                cur = predecessor[cur]
            path[node] = list(reversed(rev_path))

    return distance, path




def bellman_ford(graph, source, k):
    
    #Setting empty dictionaries for the distance, paths and relax_count
    dist = {}
    paths = {}
    relax_count = {}

    #Traversing the neighbours and setting the distance as infinity and releax count as 0 
    for node in graph.adj:
        dist[node] = float('inf')
        paths[node] = []
        relax_count[node] = 0

    #Setting our source distance to 0 and path to source as a loop . Source -> source essentially
    dist[source] = 0
    paths[source] = [source]
    
    #Our loop for the main function of finding shortest paths. We loop over the neighbours of the node until we find a shorter distance and update it.
    
    
    for _ in range(len(graph.adj) - 1):
        updated = False
        for u in graph.adj:
            for v in graph.adj[u]:
                weight = graph.w(u, v)
                if dist[u] + weight < dist[v] and relax_count[v] < k:
                    dist[v] = dist[u] + weight
                    paths[v] = paths[u] + [v]
                    #Relax count incremented with every relaxation to uphold the limit requirement
                    relax_count[v] += 1
                    updated = True
        if not updated:
            break
    
    return dist, paths