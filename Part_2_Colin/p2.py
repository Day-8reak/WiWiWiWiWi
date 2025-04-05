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
    path = {}
    relax_count = {}

    for node in graph.adj:
        distance[node] = float('inf')
        path[node] = []
        relax_count[node] = 0

    distance[source] = 0
    path[source] = [source]

    heap = graphs.MinHeap()
    heap.push((0, source, [source]))

    while not heap.is_empty():
        dist_u, u, path_u = heap.pop()

        for neighbor in graph.adjacent_nodes(u):
            weight = graph.w(u, neighbor)
            new_dist = dist_u + weight

            # Only allow relaxation if under the k-limit
            if relax_count[neighbor] < k and new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                relax_count[neighbor] += 1
                new_path = path_u + [neighbor]
                path[neighbor] = new_path
                heap.push((new_dist, neighbor, new_path))

    return distance, path



def bellman_ford(graph, source, k):
    dist = {}
    paths = {}
    relax_count = {}

    for node in graph.adj:
        dist[node] = float('inf')
        paths[node] = []
        relax_count[node] = 0

    dist[source] = 0
    paths[source] = [source]
    
    for _ in range(len(graph.adj) - 1):
        updated = False
        for u in graph.adj:
            for v in graph.adj[u]:
                weight = graph.w(u, v)
                if dist[u] + weight < dist[v] and relax_count[v] < k:
                    dist[v] = dist[u] + weight
                    paths[v] = paths[u] + [v]
                    relax_count[v] += 1
                    updated = True
        if not updated:
            break
    
    return dist, paths

































