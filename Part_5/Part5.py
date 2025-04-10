import csv  #import needed to read the csv files
import math
import matplotlib.pyplot as plt
import random
import time
import heapq
import os
import sys



#The pathing setup is just to manage and work with the imports and files better

#get the path of the parent directory as an absolute path 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Helper.graphs import DirectedWeightedGraph
#save the current path 
BASE_DIR = os.path.dirname(__file__)




#Function to read the stations from the csv file
def read_stations(filepath):
    #make an empty dictionary 
    stations = {}
    #open the csv file with newline = ' ' for line breaks. There aren't any in our csv files so this was done just in case to make the program more general
    with open(filepath, newline='') as csvfile:
        #use Dictreader from csv to make each row into its own dict with the header as the key and row value as the value
        reader = csv.DictReader(csvfile)
        #iterate over the dictionaries we made 
        for row in reader:
            #extract the value from the id key in the row dictionary i.e get the station id. 
            station_id = int(row['id'])
            #add the details of the station into our stations dictionary.
            stations[station_id] = {
                #get the station name as well as the coordinates
                'name': row['name'],
                'lat': float(row['latitude']),
                'lon': float(row['longitude'])
            }
    return stations



def haversine(lat1, lon1, lat2, lon2):  # Had to look this up online to get an accurate distance
    R = 6371  # Earth radius in kilometers
    
    # Extracting appropriate values
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    
    #Formula for distance
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def build_graph(stations, connections_file):
    # Graph class
    G = DirectedWeightedGraph()
    for station_id in stations:
        # Add stations as nodes
        G.add_node(station_id)
    # BEgin adding edges 
    with open(connections_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            s1 = int(row['station1'])
            s2 = int(row['station2'])
            # Get the true distance between the stations using our haversine function
            if s1 in stations and s2 in stations:
                lat1, lon1 = stations[s1]['lat'], stations[s1]['lon']
                lat2, lon2 = stations[s2]['lat'], stations[s2]['lon']
                dist = haversine(lat1, lon1, lat2, lon2)

                G.add_edge(s1, s2, dist)
                G.add_edge(s2, s1, dist)  # undirected
    return G



def compute_heuristic(stations, source):
    #Compute heuristic values for the stations using our distance 
    heuristic = {}
    lat1, lon1 = stations[source]['lat'], stations[source]['lon']
    for node, info in stations.items():
        heuristic[node] = haversine(lat1, lon1, info['lat'], info['lon'])
    return heuristic



def a_star(graph, source, destination, heuristic):
    
    # An A* implementation which is similar in idea to the one we made in part 4 but more suited for this question. (Had to do some studying online)
    g_score = {node: float('inf') for node in graph.adj}
    g_score[source] = 0

    predecessor = {node: None for node in graph.adj}
    #empty heap
    heap = []
    heapq.heappush(heap, (heuristic[source], source))
    #set for already visted / completed nodes
    closed = set()

    
    #While loop  for the path reconstruction and adding nodes to closed
    while heap:
        f_score, current = heapq.heappop(heap)

        if current == destination:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = predecessor[current]
            return predecessor , list(reversed(path))

        if current in closed:
            continue
        closed.add(current)

        #Checking every neighbour
        
        for neighbor in graph.adjacent_nodes(current):
            tentative_g = g_score[current] + graph.w(current, neighbor)

            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                predecessor[neighbor] = current
                f_score = tentative_g + heuristic.get(neighbor, float('inf'))
                heapq.heappush(heap, (f_score, neighbor))

    return {} , []  # No path found

#  Dijkstra using heapq 

def dijkstra_heapq(graph, source):
    distance = {node: float('inf') for node in graph.adj}
    predecessor = {node: None for node in graph.adj}
    distance[source] = 0

    heap = []
    heapq.heappush(heap, (0, source))

    while heap:
        dist_u, u = heapq.heappop(heap)

        for v in graph.adjacent_nodes(u):
            weight = graph.w(u, v)
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                heapq.heappush(heap, (distance[v], v))

    # Reconstruct full paths
    paths = {}
    for node in graph.adj:
        if distance[node] == float('inf'):
            paths[node] = []
        else:
            cur = node
            rev_path = []
            while cur is not None:
                rev_path.append(cur)
                cur = predecessor[cur]
            paths[node] = list(reversed(rev_path))

    return distance, paths

#  Main Program 
if __name__ == '__main__':
    stations = read_stations(os.path.join(BASE_DIR, 'london_stations.csv'))
    graph = build_graph(stations, os.path.join(BASE_DIR, 'london_connections.csv'))
    station_ids = list(stations.keys())

    # Single Example Run 
    source = 11
    destination = 163
    heuristic = compute_heuristic(stations, source)

    print("Running A*...")
    _ , path_a = a_star(graph, source, destination, heuristic)
    print("A* Path:", path_a)
    print("A* Length (km):", sum(graph.w(path_a[i], path_a[i+1]) for i in range(len(path_a)-1)))

    print("\nRunning Dijkstra...")
    dist_dij, paths_dij = dijkstra_heapq(graph, source)
    path_dij = paths_dij[destination]
    print("Dijkstra Path:", path_dij)
    print("Dijkstra Length (km):", sum(graph.w(path_dij[i], path_dij[i+1]) for i in range(len(path_dij)-1)))

    #Performance Comparison Experiment 
    NUM_TRIALS = 10
    results = []

    for trial in range(NUM_TRIALS):
        source, dest = random.sample(station_ids, 2)
        heuristic = compute_heuristic(stations, source)

        # A*
        start_time = time.time()
        _ , path_a = a_star(graph, source, dest, heuristic)
        time_astar = time.time() - start_time
        dist_astar = sum(graph.w(path_a[i], path_a[i+1]) for i in range(len(path_a)-1)) if len(path_a) > 1 else float('inf')

        # Dijkstra
        start_time = time.time()
        dist_dij, paths_dij = dijkstra_heapq(graph, source)
        time_dij = time.time() - start_time
        path_dij = paths_dij[dest]
        dist_dijkstra = sum(graph.w(path_dij[i], path_dij[i+1]) for i in range(len(path_dij)-1)) if len(path_dij) > 1 else float('inf')

        results.append({
            "source": source,
            "dest": dest,
            "astar_time": time_astar,
            "astar_dist": dist_astar,
            "dijkstra_time": time_dij,
            "dijkstra_dist": dist_dijkstra
        })

        print(f"Trial {trial+1}: A* Time={time_astar:.4f}s, Dijkstra Time={time_dij:.4f}s, "
              f"Dist A*={dist_astar:.2f} km, Dist Dijkstra={dist_dijkstra:.2f} km")

   
    astar_times = [r["astar_time"] for r in results]
    dijkstra_times = [r["dijkstra_time"] for r in results]
    astar_dists = [r["astar_dist"] for r in results]
    dijkstra_dists = [r["dijkstra_dist"] for r in results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(astar_times, label='A* Time', marker='o')
    plt.plot(dijkstra_times, label='Dijkstra Time', marker='x')
    plt.title("Execution Time Comparison")
    plt.xlabel("Trial")
    plt.ylabel("Time (seconds)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(astar_dists, label='A* Path Length', marker='o')
    plt.plot(dijkstra_dists, label='Dijkstra Path Length', marker='x')
    plt.title("Path Length Comparison")
    plt.xlabel("Trial")
    plt.ylabel("Distance (km)")
    plt.legend()

    plt.tight_layout()
    plt.show()





#  All-Pairs Timing Experiment

def run_all_pairs_performance_experiment(graph, stations):
    station_ids = list(stations.keys())
    total_pairs = len(station_ids) * (len(station_ids) - 1)
    dijkstra_times = []
    astar_times = []

    pair_count = 0
    print("\nRunning all-pairs timing comparison (this may take a bit)...")

    for i, source in enumerate(station_ids):
        heuristic = compute_heuristic(stations, source)

        # Dijkstra full run from source
        start_dij = time.time()
        dist_dij, paths_dij = dijkstra_heapq(graph, source)
        total_dijkstra_time = time.time() - start_dij

        valid_dests = [dest for dest in station_ids if dest != source]

        # Spread Dijkstra time across all destination lookups
        dijkstra_times.extend([total_dijkstra_time / len(valid_dests)] * len(valid_dests))

        # A* timed per destination 
        for dest in valid_dests:
            start_astar = time.time()
            a_star(graph, source, dest, heuristic)
            astar_times.append(time.time() - start_astar)

            pair_count += 1
            if pair_count % 1000 == 0:
                print(f"  Processed {pair_count}/{total_pairs} pairs...")

    # Compute Averages 
    avg_astar = sum(astar_times) / len(astar_times)
    avg_dijkstra = sum(dijkstra_times) / len(dijkstra_times)

    # Bar Plot 
    plt.figure(figsize=(8, 6))
    plt.bar(["A*", "Dijkstra"], [avg_astar, avg_dijkstra], color=["royalblue", "orange"])
    plt.ylabel("Average Execution Time (seconds)")
    plt.title("Average Execution Time for All-Pairs Shortest Paths")
    plt.grid(axis='y')
    plt.show()

    print(" Completed All-Pairs Comparison")
    print(f"A* Average Time:      {avg_astar:.8f} seconds")
    print(f"Dijkstra Average Time:{avg_dijkstra:.8f} seconds")






run_all_pairs_performance_experiment(graph, stations)
