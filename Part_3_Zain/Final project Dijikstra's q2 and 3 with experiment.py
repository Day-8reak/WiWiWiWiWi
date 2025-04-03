import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import tracemalloc

# Graph class. Copied it directly from lab 3 floyd warshall implementation
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

# Random graph generator
def generate_random_graph(num_nodes, num_edges, max_weight=10):
    G = DirectedWeightedGraph()
    for node in range(num_nodes):
        G.add_node(node)

    added_edges = 0
    attempts = 0
    max_attempts = num_nodes * num_nodes  # to prevent infinite loop

    while added_edges < num_edges and attempts < max_attempts:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and not G.are_connected(u, v):
            weight = np.random.randint(1, max_weight + 1)
            G.add_edge(u, v, weight)
            added_edges += 1
        attempts += 1

    return G


# MinHeap class with swim_up and sink_down to maximise the efficiency for dijikstras
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        self.heap.append(item)
        self.swim_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            return None
        self.swap(0, len(self.heap) - 1)
        item = self.heap.pop()
        self.sink_down(0)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def swim_up(self, index):
        parent = (index - 1) // 2
        while index > 0 and self.heap[index][0] < self.heap[parent][0]:
            self.swap(index, parent)
            index = parent
            parent = (index - 1) // 2

    def sink_down(self, index):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2

        if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
            smallest = left
        if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
            smallest = right

        if smallest != index:
            self.swap(index, smallest)
            self.sink_down(smallest)

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

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

    heap = MinHeap()
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







G = DirectedWeightedGraph()
for node in range(4):
    G.add_node(node)

G.add_edge(0, 1, 4)
G.add_edge(0, 2, 1)
G.add_edge(2, 1, 2)
G.add_edge(1, 3, 1)
G.add_edge(2, 3, 5)

distances, paths = dijkstra(G, 0, 2)

print("Distances:", distances)
print("Paths:", paths)






# All-pairs shortest path using Dijkstra
def all_pairs_dijkstra(graph, k):
    all_distances = {}
    all_paths = {}

    for source in graph.adj:
        distances, paths = dijkstra(graph, source, k)
        all_distances[source] = distances
        all_paths[source] = paths

    return all_distances, all_paths




G = DirectedWeightedGraph()
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














#  EXPERIMENTS 

# Test with variable nodes, fixed number of edges
def experiment_variable_nodes_fixed_edges():
    node_sizes = [10, 30, 50, 100, 200]
    fixed_edges = 300

    single_times = []
    allpair_times = []

    for V in node_sizes:
        G = generate_random_graph(V, fixed_edges)
        
        start = time.time()
        dijkstra(G, 0, k=V-1)
        end = time.time()
        single_times.append((end - start) * 1000)

        start = time.time()
        all_pairs_dijkstra(G, k=V-1)
        end = time.time()
        allpair_times.append((end - start) * 1000)

    plt.figure()
    plt.plot(node_sizes, single_times, label="Single-Source", marker='o')
    plt.plot(node_sizes, allpair_times, label="All-Pairs", marker='s')
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

    single_times = []
    allpair_times = []

    for E in edge_counts:
        G = generate_random_graph(fixed_nodes, E)
        
        start = time.time()
        dijkstra(G, 0, k=fixed_nodes-1)
        end = time.time()
        single_times.append((end - start) * 1000)

        start = time.time()
        all_pairs_dijkstra(G, k=fixed_nodes-1)
        end = time.time()
        allpair_times.append((end - start) * 1000)

    plt.figure()
    plt.plot(edge_counts, single_times, label="Single-Source", marker='o')
    plt.plot(edge_counts, allpair_times, label="All-Pairs", marker='s')
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
    single_times = []
    allpair_times = []

    for V, E in configs:
        G = generate_random_graph(V, E)

        start = time.time()
        dijkstra(G, 0, k=V-1)
        end = time.time()
        single_times.append((end - start) * 1000)

        start = time.time()
        all_pairs_dijkstra(G, k=V-1)
        end = time.time()
        allpair_times.append((end - start) * 1000)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, single_times, width, label='Single-Source')
    plt.bar(x + width/2, allpair_times, width, label='All-Pairs')

    plt.xticks(x, labels)
    plt.title("Small vs Medium vs Large Graphs")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()












def experiment_accuracy_vs_k():
    print("Running accuracy check against variable k")

    V = 300
    E = 1000
    G = generate_random_graph(V, E)

    source = 0
    k_values = [0, 1, 2, 3, 5, 10, 20, 50, V - 1]
    accuracies = []

    # Baseline: k = V - 1
    baseline_distances, _ = dijkstra(G, source, V - 1)

    # Find all nodes that are reachable in the baseline
    reachable_nodes = [node for node in G.adj if node != source and baseline_distances[node] != float('inf')]
    total_reachable = len(reachable_nodes)
    
    
    
    if total_reachable == 0:
        print(" No nodes are reachable from the source in the baseline run. Try increasing edge count.")
        return

    for k in k_values:
        distances, _ = dijkstra(G, source, k)

        correct = 0
        for node in reachable_nodes:
            if distances[node] == baseline_distances[node]:
                correct += 1

        accuracy = (correct / total_reachable) * 100 if total_reachable > 0 else 0
        accuracies.append(accuracy)
        print(f"k={k}: Accuracy = {accuracy:.2f}% (Correct={correct}, Reachable={total_reachable})")

    # Plot
    plt.figure()
    plt.plot(k_values, accuracies, marker='^', color='green')
    plt.title("Accuracy vs Relaxation Limit (k)")
    plt.xlabel("k (Relaxation Limit)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.show()












def experiment_space_vs_nodes_fixed_edges():
    
    
    
    
    print("Running space check for nodes vs fixed edges")
    node_sizes = [10, 30, 50, 100, 200]
    fixed_edges = 300
    memory_usages = []

    for V in node_sizes:
        G = generate_random_graph(V, fixed_edges)

        tracemalloc.start()
        dijkstra(G, 0, k=V-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_usages.append(peak / 1024)  # Convert to KB

    plt.figure()
    plt.plot(node_sizes, memory_usages, marker='o', color='purple')
    plt.title("Memory Usage vs Nodes (Fixed {} Edges)".format(fixed_edges))
    plt.xlabel("Number of Nodes")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.grid(True)
    plt.show()




def experiment_space_vs_edges_fixed_nodes():
    
    
    print("Running space check for edges vs fixed nodes")
    fixed_nodes = 100
    edge_counts = [100, 500, 1000, 2000, 4000]
    memory_usages = []

    for E in edge_counts:
        G = generate_random_graph(fixed_nodes, E)

        tracemalloc.start()
        dijkstra(G, 0, k=fixed_nodes-1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_usages.append(peak / 1024)  # Convert to KB

    plt.figure()
    plt.plot(edge_counts, memory_usages, marker='s', color='brown')
    plt.title("Memory Usage vs Edges (Fixed {} Nodes)".format(fixed_nodes))
    plt.xlabel("Number of Edges")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.grid(True)
    plt.show()






































# Run all experiments
#experiment_variable_edges_fixed_nodes()
#experiment_small_vs_large_graphs()
#experiment_space_vs_nodes_fixed_edges()
#experiment_space_vs_edges_fixed_nodes()
experiment_accuracy_vs_k()


G = DirectedWeightedGraph()
for i in range(5):
    G.add_node(i)
for i in range(4):
    G.add_edge(i, i+1, 1)

d1, _ = dijkstra(G, 0, 1)
d2, _ = dijkstra(G, 0, 4)

print("Distances with k=1:", d1)
print("Distances with k=4:", d2)