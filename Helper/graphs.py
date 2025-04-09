import numpy as np

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