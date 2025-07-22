import networkx as nx
import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, csr_matrix
from math_satellites import generate_synthetic_satellite_grid
from math_network_setup import (
    generate_sessions,
    assign_users_to_closest_satellites,
    plot_colored_user_satellite_graph,
    print_user_satellite_pairs,
    print_satellite_connectivity
)

def build_system_matrix(graph):
    nodes = list(graph.nodes)
    idx_map = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)

    A = lil_matrix((N, N))
    for node in nodes:
        i = idx_map[node]
        neighbors = list(graph.neighbors(node))
        A[i, i] = len(neighbors)
        for nbr in neighbors:
            j = idx_map[nbr]
            A[i, j] = -1

    return csr_matrix(A), idx_map, nodes

def solve_flow_jacobi_sparse(A, idx_map, nodes, source, target, max_iter=100, tol=1e-4):
    N = len(nodes)
    b = np.zeros(N)
    b[idx_map[source]] = 1
    b[idx_map[target]] = -1

    x = np.zeros(N)
    A_diag = A.diagonal()
    A_diag_inv = 1.0 / A_diag

    for _ in range(max_iter):
        Ax = A @ x
        x_new = x + A_diag_inv * (b - Ax)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new

    return {node: x[idx_map[node]] for node in nodes}

def compute_satellite_routes_jacobi(users, user_to_satellite, satellite_positions, connectivity):
    results = {}
    satellite_graph = nx.Graph()
    for u in connectivity:
        for v in connectivity[u]:
            satellite_graph.add_edge(u, v)

    A, idx_map, nodes = build_system_matrix(satellite_graph)

    for user1, user2 in combinations(users, 2):
        uid1, uid2 = user1.get_id(), user2.get_id()
        sat1, sat2 = user_to_satellite[uid1], user_to_satellite[uid2]

        if nx.has_path(satellite_graph, sat1, sat2):
            flow = solve_flow_jacobi_sparse(A, idx_map, nodes, source=sat1, target=sat2)
            total_flow = sum(abs(f) for f in flow.values())
            results[(uid1, uid2)] = {
                "flow": flow,
                "total_flow": total_flow
            }
        else:
            results[(uid1, uid2)] = {
                "flow": None,
                "total_flow": float("inf")
            }
    return results

if __name__ == "__main__":
    num_satellites = 50
    num_sessions = 1
    users_per_session = 10

    satellite_positions, connectivity = generate_synthetic_satellite_grid(
        num_satellites, lat_range=(30, 55), lon_range=(-140, 160))

    sessions = generate_sessions(num_sessions, users_per_session)
    session = sessions[0]
    users = session.get_user()

    user_to_satellite_map = assign_users_to_closest_satellites(users, satellite_positions)
    print_user_satellite_pairs(users, user_to_satellite_map)
    print_satellite_connectivity(connectivity)



    print("\nUser Pair Routing via Optimized Jacobi Method:")
    results = compute_satellite_routes_jacobi(users, user_to_satellite_map, satellite_positions, connectivity)

    for (u1, u2), data in results.items():
        flow = data["flow"]
        if flow:
            significant = {s: f for s, f in flow.items() if abs(f) > 0.01}
            print(f"User {u1} <-> User {u2}: Flow Path = [" + ", ".join(f"S{s}:{f:.2f}" for s, f in significant.items()) + "]")
        else:
            print(f"User {u1} <-> User {u2}: No valid flow path")

    plot_colored_user_satellite_graph(session.get_user(), satellite_positions, user_to_satellite_map)
