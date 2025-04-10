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
import Part_2_Colin.p2 as p2




G = graphs.DirectedWeightedGraph()
for node in range(4):
    G.add_node(node)

G.add_edge(0, 1, 4)
G.add_edge(0, 2, 1)
G.add_edge(2, 1, 2)
G.add_edge(1, 3, 1)
G.add_edge(2, 3, 5)

distances, paths = p2.dijkstra(G, 0, 2)

print("Distances:", distances)
print("Paths:", paths)



# All-pairs shortest path using Dijkstra
def all_pairs_dijkstra(graph, k):
    all_distances = {}
    all_paths = {}

    for source in graph.adj:
        distances, paths = p2.dijkstra(graph, source, k)
        all_distances[source] = distances
        all_paths[source] = paths

    return all_distances, all_paths

# All- paris shortest using Bellman-Ford
def all_pairs_bf(graph, k):
    all_distances = {}
    all_paths = {}

    for source in graph.adj:
        distances, paths = p2.bellman_ford(graph, source, k)
        all_distances[source] = distances
        all_paths[source] = paths

    return all_distances, all_paths


G = graphs.DirectedWeightedGraph()
for node in range(4):
    G.add_node(node)

G.add_edge(0, 1, 4)
G.add_edge(0, 2, 1)
G.add_edge(2, 1, 2)
G.add_edge(1, 3, 1)
G.add_edge(2, 3, 5)

all_distances, all_paths = all_pairs_dijkstra(G, k=2)

print("All-Pairs Distances:")
for u in all_distances:
    print(f"From {u}: {all_distances[u]}")

print("\nAll-Pairs Paths:")
for u in all_paths:
    print(f"From {u}: {all_paths[u]}")


all_distances, all_paths = all_pairs_bf(G, k=2)

print("All-Pairs Distances:")
for u in all_distances:
    print(f"From {u}: {all_distances[u]}")

print("\nAll-Pairs Paths:")
for u in all_paths:
    print(f"From {u}: {all_paths[u]}")





def experiment_all_pairs_performance():
    # Part 1: Varying nodes, fixed density
    node_sizes = [10, 20, 30, 40, 50]
    edge_density = 0.8
    dijkstra_times = []
    bellman_times = []

    for V in node_sizes:
        max_edges = V * (V - 1)
        E = int(edge_density * max_edges)
        G = graphs.generate_random_graph(V, E)

        start = time.time()
        all_pairs_dijkstra(G, k=V - 1)
        dijkstra_times.append((time.time() - start) * 1000)

        start = time.time()
        all_pairs_bf(G, k=V - 1)
        bellman_times.append((time.time() - start) * 1000)

    plt.figure()
    plt.plot(node_sizes, dijkstra_times, marker='o', label='All-Pairs Dijkstra')
    plt.plot(node_sizes, bellman_times, marker='s', label='All-Pairs Bellman-Ford')
    plt.title("All-Pairs: Varying Nodes (80% Dense)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Part 2: Fixed nodes, varying edges
    fixed_nodes = 50
    edge_counts = [100, 500, 1000, 1500, 2000]
    dijkstra_times = []
    bellman_times = []

    for E in edge_counts:
        G = graphs.generate_random_graph(fixed_nodes, E)

        start = time.time()
        all_pairs_dijkstra(G, k=fixed_nodes - 1)
        dijkstra_times.append((time.time() - start) * 1000)

        start = time.time()
        all_pairs_bf(G, k=fixed_nodes - 1)
        bellman_times.append((time.time() - start) * 1000)

    plt.figure()
    plt.plot(edge_counts, dijkstra_times, marker='o', label='All-Pairs Dijkstra')
    plt.plot(edge_counts, bellman_times, marker='s', label='All-Pairs Bellman-Ford')
    plt.title("All-Pairs: Fixed Nodes (V=50), Varying Edges")
    plt.xlabel("Number of Edges")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()



#All pairs performance 
experiment_all_pairs_performance()
"""
G = DirectedWeightedGraph()
for i in range(5):
    G.add_node(i)
for i in range(4):
    G.add_edge(i, i+1, 1)

d1, _ = dijkstra(G, 0, 1)
d2, _ = dijkstra(G, 0, 4)

print("Distances with k=1:", d1)
print("Distances with k=4:", d2)
"""