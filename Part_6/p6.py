import Helper.graphs as graphs
import heapq

# Class responsible for finding shortest paths using a selected algorithm
class ShortPathFinder:
    def __init__(self):
        return

    # Method to calculate the shortest path between source and destination
    def calc_short_path(self, source:int, dest:int):
        return self.algorithm(self.graph, source, dest)
    
    # Method to set the graph for the algorithm
    def set_graph(self, graph):
        self.graph = graph

    # Method to set the algorithm to be used (e.g., Dijkstra, Bellman-Ford, A*)
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm


# Dijkstra's Algorithm for shortest path finding
class Dijkstra:
    def __init__(self):
        return
    
    # Method to calculate the shortest path using Dijkstra's algorithm
    def calc_sp(self, graph, source, dest):
        distance = {}         # Dictionary to store the shortest distance from source to each node
        predecessor = {}      # Dictionary to store the predecessor of each node for path reconstruction
        relax_count = {}      # Dictionary to count the number of relaxations for each node (unused in Dijkstra)

        # Initialize distance to infinity and predecessor to None for all nodes
        for node in graph.adj:
            distance[node] = float('inf')
            predecessor[node] = None
            relax_count[node] = 0

        distance[source] = 0  # Distance from source to itself is 0

        # Min-heap (priority queue) for selecting the node with the smallest distance
        heap = graphs.MinHeap()
        heap.push((0, source))  # Start by pushing the source node with a distance of 0

        # Main loop of Dijkstra's algorithm
        while not heap.is_empty():
            dist_u, u = heap.pop()  # Pop the node with the smallest distance

            # Update the distances for each neighbor of u
            for neighbor in graph.adjacent_nodes(u):
                weight = graph.w(u, neighbor)  # Weight of the edge
                new_dist = dist_u + weight  # Potential new distance

                # Relaxation condition
                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    predecessor[neighbor] = u
                    heap.push((new_dist, neighbor))

        # Reconstruct the shortest path from source to destination using the predecessor dictionary
        path = {}
        for node in graph.adj:
            if distance[node] == float('inf'):
                path[node] = []  # No path
            else:
                cur = node
                rev_path = []
                while cur is not None:
                    rev_path.append(cur)
                    cur = predecessor[cur]
                path[node] = list(reversed(rev_path))

        return distance[dest]  # Return the distance to the destination node


# Bellman-Ford Algorithm for shortest path finding (can handle negative weights)
class Bellman_Ford:
    def __init__(self):
        return
    
    # Method to calculate the shortest path using Bellman-Ford's algorithm
    def calc_sp(self, graph, source, dest):
        dist = {}         # Dictionary to store the shortest distance from source to each node
        paths = {}        # Dictionary to store the path to each node
        relax_count = {}  # Dictionary to count the number of relaxations for each node (unused in Bellman-Ford)

        # Initialize distance to infinity and paths to empty list for all nodes
        for node in graph.adj:
            dist[node] = float('inf')
            paths[node] = []
            relax_count[node] = 0

        dist[source] = 0  # Distance from source to itself is 0
        paths[source] = [source]  # Path from source to itself is just the source

        # Perform relaxations for |V| - 1 times (where |V| is the number of vertices)
        for _ in range(len(graph.adj) - 1):
            updated = False  # Flag to track if any update occurs in this iteration
            # Traverse all edges and perform relaxation
            for u in graph.adj:
                for v in graph.adj[u]:
                    weight = graph.w(u, v)
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        paths[v] = paths[u] + [v]  # Update the path
                        updated = True
            if not updated:
                break  # If no update happens, break early

        return dist[dest]  # Return the distance to the destination node


# A* Algorithm for shortest path finding (uses a heuristic to guide the search)
class A_Star:
    def __init__(self):
        return
    
    # Method to calculate the shortest path using A* algorithm
    def calc_sp(self, graph, source, dest):
        g_score = {node: float('inf') for node in graph.adj}  # g(n): actual cost from source to node
        g_score[source] = 0  # g(source) is 0

        predecessor = {node: None for node in graph.adj}  # To store the predecessor of each node

        heap = []  # Min-heap (priority queue) for selecting the node with the smallest f-score
        heapq.heappush(heap, (graph.get_heuristic()[source], source))  # Push the source with heuristic as f-score

        closed = set()  # Set to track visited nodes

        # Main loop of A* algorithm
        while heap:
            f_score, current = heapq.heappop(heap)  # Pop the node with the smallest f-score

            if current == dest:
                # Reconstruct the shortest path
                path = []
                dist = 0
                while current is not None:
                    path.append(current)
                    current = predecessor[current]

                path.reverse()

                # Calculate the total distance by summing up the weights of edges in the path
                for x in range(len(path) - 1):
                    dist += graph.w(path[x], path[x + 1])

                return dist  # Return the total distance

            if current in closed:
                continue  # Skip if the node has already been visited
            closed.add(current)

            # Update the neighbors of the current node
            for neighbor in graph.adjacent_nodes(current):
                tentative_g = g_score[current] + graph.w(current, neighbor)

                # If a shorter path to the neighbor is found, update the g-score and predecessor
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    predecessor[neighbor] = current
                    f_score = tentative_g + graph.get_heuristic().get(neighbor, float('inf'))
                    heapq.heappush(heap, (f_score, neighbor))  # Push the updated neighbor to the heap

        return None  # Return None if no path is found


# Weighted Graph class to represent a graph with weighted edges
class WeightedGraph:

    def __init__(self):
        self.adj = {}      # Dictionary to store adjacency list (nodes and their neighbors)
        self.weights = {}  # Dictionary to store edge weights

    # Method to get adjacent nodes of a specific node
    def get_adj_nodes(self, node):
        return self.adj[node]

    # Method to add a new node to the graph
    def add_node(self, node):
        self.adj[node] = []  # Initialize the adjacency list for this node

    # Method to add an edge between two nodes with a specified weight
    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1] and node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)
            self.weights[(node1, node2)] = weight

    # Method to get the number of nodes in the graph
    def get_num_of_nodes(self):
        return len(self.adj)

    # Method to get the weight of an edge between two nodes
    def w(self, node1, node2):
        if (node1, node2) in self.weights.keys():
            return self.weights[(node1, node2)]
        else:
            return None  # Return None if the edge doesn't exist


# Heuristic Graph class that extends WeightedGraph and adds heuristic functionality for A*
class HeuristicGraph(WeightedGraph):
    def __init__(self, heuristic):
        self.heuristic = heuristic  # Heuristic is manually provided upon instantiation
        return
    
    # Method to get the heuristic values for the nodes
    def get_heuristic(self):
        return self.heuristic
