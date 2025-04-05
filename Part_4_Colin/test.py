import heapq

def A_Star(graph, source, destination, heuristic):
    # The open set as a priority queue
    open_set = []
    # Each element in the queue is a tuple: (f_score, current_node)
    heapq.heappush(open_set, (heuristic[source], source))
    
    # Dictionaries to track the best cost so far (g score) and the best predecessor for each node.
    g_score = {node: float('inf') for node in graph}
    g_score[source] = 0
    
    predecessors = {node: None for node in graph}
    
    # To keep track of nodes that have been processed
    closed_set = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # If we reached the destination, reconstruct the path and return.
        if current == destination:
            path = reconstruct_path(predecessors, source, destination)
            return (predecessors, path)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        # Check all the neighbors of the current node.
        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one.
                g_score[neighbor] = tentative_g_score
                predecessors[neighbor] = current
                # f = g + heuristic
                f_score = tentative_g_score + heuristic.get(neighbor, 0)
                heapq.heappush(open_set, (f_score, neighbor))
                
    # If we get here, no path was found.
    return (predecessors, [])

def reconstruct_path(predecessors, source, destination):
    """Utility function to reconstruct the path from source to destination."""
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

# Example usage:
if __name__ == '__main__':
    # Example graph: node -> {neighbor: weight, ...}
    graph = {
        1: {2: 1, 3: 4},
        2: {3: 2, 4: 5},
        3: {4: 1},
        4: {7: 0},
        5: {6: 1},
        6: {7: 2},
        7: {10: 3},
        8: {9: 1},
        9: {10: 2},
        10: {}
    }
    
    # Example heuristic: assume an underestimate of the distance to node 4.
    heuristic = {
        1: 9,
        2: 8,
        3: 7,
        4: 4,
        5: 6,
        6: 5, 
        7: 3, 
        8: 2,
        9: 1,
        10: 0
    }
    
    pred, shortest_path = A_Star(graph, 1, 10, heuristic)
    print("Predecessors:", pred)
    print("Shortest path from 1 to 10:", shortest_path)
