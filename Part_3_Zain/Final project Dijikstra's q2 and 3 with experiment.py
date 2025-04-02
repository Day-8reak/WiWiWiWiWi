import time
import numpy as np
import copy
import matplotlib.pyplot as plt

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

        if relax_count[u] >= k:
            continue
        relax_count[u] += 1

        for neighbor in graph.adjacent_nodes(u):
            weight = graph.w(u, neighbor)
            new_dist = dist_u + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
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















# Experiment for single-source Dijkstra
def experiment_single_source():
    sizes = [10, 50, 100, 200]
    times = []

    for V in sizes:
        G = generate_random_graph(V, density=0.2)
        start = time.time()
        dijkstra(G, 0, k=V-1)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds

    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.title("Dijkstra Single-Source Time vs Graph Size")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.show()

# Experiment for all-pairs Dijkstra
def experiment_all_pairs():
    sizes = [10, 30, 50, 70]
    times = []

    for V in sizes:
        G = generate_random_graph(V, density=0.2)
        start = time.time()
        all_pairs_dijkstra(G, k=V-1)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds

    plt.figure()
    plt.plot(sizes, times, marker='s', color='orange')
    plt.title("All-Pairs Dijkstra Time vs Graph Size")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.show()

# Run the experiments
experiment_single_source()
experiment_all_pairs()
