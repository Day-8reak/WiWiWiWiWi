import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import tracemalloc
import p2

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import Helper.graphs as graphs


#  EXPERIMENTS 

# Test with variable nodes, fixed number of edges
def experiment_variable_nodes_fixed_edges():
    node_sizes = [10, 30, 50, 100, 200]
    fixed_edges = 300

    dijkstra_times = []
    bf_times = []
    dijkstra_times_k2 = []
    bf_times_k2 = []

    for V in node_sizes:
        G = graphs.generate_random_graph(V, fixed_edges)
        
        # Dijkstra with k = V-1
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=V-1)
        end = time.perf_counter()
        dijkstra_times.append((end - start) * 1000)

        # Bellman-Ford with k = V-1
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=V-1)
        end = time.perf_counter()
        bf_times.append((end - start) * 1000)

        # Dijkstra with k = 2
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=2)
        end = time.perf_counter()
        dijkstra_times_k2.append((end - start) * 1000)

        # Bellman-Ford with k = 2
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=2)
        end = time.perf_counter()
        bf_times_k2.append((end - start) * 1000)

    # Plotting the results
    plt.figure()
    plt.plot(node_sizes, dijkstra_times, label="Dijkstra (k = V-1)", marker='o')
    plt.plot(node_sizes, bf_times, label="Bellman-Ford (k = V-1)", marker='s')
    plt.plot(node_sizes, dijkstra_times_k2, label="Dijkstra (k = 2)", marker='x')
    plt.plot(node_sizes, bf_times_k2, label="Bellman-Ford (k = 2)", marker='^')
    plt.title("Variable Nodes, Fixed Edges ({} edges)".format(fixed_edges))
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Test with variable edges, fixed number of nodes
def experiment_variable_edges_fixed_nodes():
    fixed_nodes = 100
    edge_counts = [100, 500, 1000, 2000, 4000]

    dijkstra_times = []
    bf_times = []
    dijkstra_times_k2 = []
    bf_times_k2 = []

    for E in edge_counts:
        G = graphs.generate_random_graph(fixed_nodes, E)
        
         # Dijkstra with k = V-1
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=fixed_nodes-1)
        end = time.perf_counter()
        dijkstra_times.append((end - start) * 1000)

        # Bellman-Ford with k = V-1
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=fixed_nodes-1)
        end = time.perf_counter()
        bf_times.append((end - start) * 1000)

        # Dijkstra with k = 2
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=2)
        end = time.perf_counter()
        dijkstra_times_k2.append((end - start) * 1000)

        # Bellman-Ford with k = 2
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=2)
        end = time.perf_counter()
        bf_times_k2.append((end - start) * 1000)

    plt.figure()
    plt.plot(edge_counts, dijkstra_times, label="Dijkstra (k = V-1)", marker='o')
    plt.plot(edge_counts, bf_times, label="Bellman-Ford (k = V-1)", marker='s')
    plt.plot(edge_counts, dijkstra_times_k2, label="Dijkstra (k = 2)", marker='x')
    plt.plot(edge_counts, bf_times_k2, label="Bellman-Ford (k = 2)", marker='^')
    plt.title("Variable Edges, Fixed Nodes ({} nodes)".format(fixed_nodes))
    plt.xlabel("Number of Edges")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Compare performance on small vs large graphs
def experiment_small_vs_large_graphs():
    configs = [
        (10, 30),   # Small
        (50, 500),  # Medium
        (200, 2000) # Large
    ]

    labels = ["Small", "Medium", "Large"]
    dijkstra_times = []
    bf_times = []
    dijkstra_times_k2 = []
    bf_times_k2 = []

    for V, E in configs:
        G = graphs.generate_random_graph(V, E)

        # Dijkstra with k = V-1
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=V-1)
        end = time.perf_counter()
        dijkstra_times.append((end - start) * 1000)

        # Bellman-Ford with k = V-1
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=V-1)
        end = time.perf_counter()
        bf_times.append((end - start) * 1000)

        # Dijkstra with k = 2
        start = time.perf_counter()
        p2.dijkstra(G, 0, k=2)
        end = time.perf_counter()
        dijkstra_times_k2.append((end - start) * 1000)

        # Bellman-Ford with k = 2
        start = time.perf_counter()
        p2.bellman_ford(G, 0, k=2)
        end = time.perf_counter()
        bf_times_k2.append((end - start) * 1000)
    x = np.arange(len(labels))
    width = 0.2

    plt.figure()
    # Plot bars for each k (k=V-1 and k=2)
    plt.bar(x - width, dijkstra_times, width, label='Dijkstra (k=V-1)')
    plt.bar(x, dijkstra_times_k2, width, label='Dijkstra (k=2)')
    plt.bar(x + width, bf_times, width, label='Bellman-Ford (k=V-1)')
    plt.bar(x + 2 * width, bf_times_k2, width, label='Bellman-Ford (k=2)')

    plt.xticks(x, labels)
    plt.title("Small vs Medium vs Large Graphs with Different k Values")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()





def experiment_accuracy_vs_k():
    print("Running accuracy check against variable k")

    V = 50
    E = 500
    G = graphs.generate_random_graph(V, E)

    source = 0
    k_values = [0, 1, 2, 3, 5, 10, 20, 50, V - 1]
    dijkstra_accuracies = []
    bf_accuracies = []

    # Baseline: k = V - 1
    baseline_distances, _ = p2.dijkstra(G, source, V - 1)

    # Find all nodes that are reachable in the baseline
    reachable_nodes = [node for node in G.adj if node != source and baseline_distances[node] != float('inf')]
    total_reachable = len(reachable_nodes)

    if total_reachable == 0:
        print("No nodes are reachable from the source in the baseline run. Try increasing edge count.")
        return

    # Accuracy for Dijkstra
    for k in k_values:
        distances, _ = p2.dijkstra(G, source, k)

        correct = 0
        for node in reachable_nodes:
            if distances[node] == baseline_distances[node]:
                correct += 1

        accuracy = (correct / total_reachable) * 100 if total_reachable > 0 else 0
        dijkstra_accuracies.append(accuracy)
        print(f"Dijkstra k={k}: Accuracy = {accuracy:.2f}% (Correct={correct}, Reachable={total_reachable})")

    # Accuracy for Bellman-Ford
    for k in k_values:
        distances, _ = p2.bellman_ford(G, source, k)

        correct = 0
        for node in reachable_nodes:
            if distances[node] == baseline_distances[node]:
                correct += 1

        accuracy = (correct / total_reachable) * 100 if total_reachable > 0 else 0
        bf_accuracies.append(accuracy)
        print(f"Bellman-Ford k={k}: Accuracy = {accuracy:.2f}% (Correct={correct}, Reachable={total_reachable})")

    # Plot
    plt.figure()
    plt.plot(k_values, dijkstra_accuracies, marker='^', label="Dijkstra", color='green')
    plt.plot(k_values, bf_accuracies, marker='o', label="Bellman-Ford", color='blue')
    plt.title("Accuracy vs Relaxation Limit (k)")
    plt.xlabel("k (Relaxation Limit)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.show()












def experiment_space_vs_nodes_fixed_edges():
    
    print("Running space check for nodes vs fixed edges")
    node_sizes = [10, 30, 50, 100, 200]
    fixed_edges = 300
    memory_usages_dijkstra = []
    memory_usages_bf = []
    memory_usages_dijkstra_k2 = []
    memory_usages_bf_k2 = []

    for V in node_sizes:
        G = graphs.generate_random_graph(V, fixed_edges)

        # Dijkstra with k = V-1
        tracemalloc.start()
        p2.dijkstra(G, 0, k=V-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_dijkstra.append(peak / 1024)  # Convert to KB

        # Bellman-Ford with k = V-1
        tracemalloc.start()
        p2.bellman_ford(G, 0, k=V-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_bf.append(peak / 1024)  # Convert to KB

        # Dijkstra with k = 2
        tracemalloc.start()
        p2.dijkstra(G, 0, k=2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_dijkstra_k2.append(peak / 1024)  # Convert to KB

        # Bellman-Ford with k = 2
        tracemalloc.start()
        p2.bellman_ford(G, 0, k=2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_bf_k2.append(peak / 1024)  # Convert to KB
    
    # Plotting results
    plt.figure()

    plt.plot(node_sizes, memory_usages_dijkstra, marker='o', label="Dijkstra (k=V-1)", color='blue')
    plt.plot(node_sizes, memory_usages_bf, marker='s', label="Bellman-Ford (k=V-1)", color='green')
    plt.plot(node_sizes, memory_usages_dijkstra_k2, marker='^', label="Dijkstra (k=2)", color='red')
    plt.plot(node_sizes, memory_usages_bf_k2, marker='d', label="Bellman-Ford (k=2)", color='orange')

    plt.title("Memory Usage vs Nodes (Fixed {} Edges)".format(fixed_edges))
    plt.xlabel("Number of Nodes")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.legend()
    plt.grid(True)
    plt.show()




def experiment_space_vs_edges_fixed_nodes():
    
    
    print("Running space check for edges vs fixed nodes")
    fixed_nodes = 100
    edge_counts = [100, 500, 1000, 2000, 4000]
    memory_usages_dijkstra = []
    memory_usages_bf = []
    memory_usages_dijkstra_k2 = []
    memory_usages_bf_k2 = []

    for E in edge_counts:
        G = graphs.generate_random_graph(fixed_nodes, E)

        # Dijkstra with k = V-1
        tracemalloc.start()
        p2.dijkstra(G, 0, k=fixed_nodes-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_dijkstra.append(peak / 1024)  # Convert to KB

        # Bellman-Ford with k = V-1
        tracemalloc.start()
        p2.bellman_ford(G, 0, k=fixed_nodes-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_bf.append(peak / 1024)  # Convert to KB

        # Dijkstra with k = 2
        tracemalloc.start()
        p2.dijkstra(G, 0, k=2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_dijkstra_k2.append(peak / 1024)  # Convert to KB

        # Bellman-Ford with k = 2
        tracemalloc.start()
        p2.bellman_ford(G, 0, k=2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usages_bf_k2.append(peak / 1024)  # Convert to KB

    # Plotting the results
    plt.figure()

    plt.plot(edge_counts, memory_usages_dijkstra, marker='o', label="Dijkstra (k=V-1)", color='blue')
    plt.plot(edge_counts, memory_usages_bf, marker='s', label="Bellman-Ford (k=V-1)", color='green')
    plt.plot(edge_counts, memory_usages_dijkstra_k2, marker='^', label="Dijkstra (k=2)", color='red')
    plt.plot(edge_counts, memory_usages_bf_k2, marker='d', label="Bellman-Ford (k=2)", color='orange')

    plt.title("Memory Usage vs Edges (Fixed {} Nodes)".format(fixed_nodes))
    plt.xlabel("Number of Edges")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.legend()
    plt.grid(True)
    plt.show()




    










# Run all experiments
experiment_variable_nodes_fixed_edges()
experiment_variable_edges_fixed_nodes()
experiment_small_vs_large_graphs()
experiment_space_vs_nodes_fixed_edges()
experiment_space_vs_edges_fixed_nodes()
experiment_accuracy_vs_k()

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
