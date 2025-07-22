import networkx as nx
import numpy as np
from itertools import combinations
from math_satellites import generate_synthetic_satellite_grid, calculate_latency
from math_network_setup import (
    generate_sessions,
    assign_users_to_closest_satellites,
    plot_colored_user_satellite_graph,
    print_user_satellite_pairs,
    print_satellite_connectivity
)

SATELLITE_CAPACITY = 20

def build_satellite_graph(satellite_positions, connectivity):
    G = nx.Graph()
    for u in connectivity:
        for v in connectivity[u]:
            if not G.has_edge(u, v):
                lat1, lon1 = satellite_positions[u]
                lat2, lon2 = satellite_positions[v]
                latency = calculate_latency(lat1, lon1, lat2, lon2)
                G.add_edge(u, v, weight=latency)
    return G

def compute_satellite_routes_dijkstra(users, user_to_satellite, satellite_positions, connectivity):
    results = {}
    G = build_satellite_graph(satellite_positions, connectivity)
    satellite_usage = {i: 0 for i in range(len(satellite_positions))}

    for user1, user2 in combinations(users, 2):
        uid1, uid2 = user1.get_id(), user2.get_id()
        sat1, sat2 = user_to_satellite[uid1], user_to_satellite[uid2]

        if nx.has_path(G, sat1, sat2):
            path = nx.dijkstra_path(G, source=sat1, target=sat2, weight="weight")
            latency = nx.dijkstra_path_length(G, sat1, sat2, weight="weight")
            for s in path:
                satellite_usage[s] += 1
            results[(uid1, uid2)] = {
                "latency": latency,
                "path": path
            }
        else:
            results[(uid1, uid2)] = {
                "latency": float('inf'),
                "path": None
            }

    return results

if __name__ == "__main__":
    num_satellites = 100
    num_sessions = 1
    users_per_session = 20

    satellite_positions, connectivity = generate_synthetic_satellite_grid(
        num_satellites, lat_range=(30, 55), lon_range=(-140, 160))

    sessions = generate_sessions(num_sessions, users_per_session)
    session = sessions[0]
    users = session.get_user()

    user_to_satellite_map = assign_users_to_closest_satellites(users, satellite_positions)

    print_user_satellite_pairs(users, user_to_satellite_map)
    print_satellite_connectivity(connectivity)



    print("\nUser Pair Routing via Dijkstra Method:")
    results = compute_satellite_routes_dijkstra(users, user_to_satellite_map, satellite_positions, connectivity)

    for (u1, u2), data in results.items():
        if data["path"]:
            readable_path = " -> ".join(f"S{s}" for s in data["path"])
            print(f"User {u1} <-> User {u2}: Latency = {data['latency']:.2f} ms, Path = [{readable_path}]")
        else:
            print(f"User {u1} <-> User {u2}: No valid path found")

    plot_colored_user_satellite_graph(session.get_user(), satellite_positions, user_to_satellite_map)