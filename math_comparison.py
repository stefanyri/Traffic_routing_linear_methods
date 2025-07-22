import time
import numpy as np
import matplotlib.pyplot as plt
from math_satellites import generate_synthetic_satellite_grid, calculate_latency
from math_network_setup import (
    generate_sessions,
    assign_users_to_closest_satellites,
    print_user_satellite_pairs,
    print_satellite_connectivity
)
from math_jacobi import compute_satellite_routes_jacobi
from math_gauss_seidel import compute_satellite_routes_gauss_seidel
from math_greedy import compute_satellite_routes_dijkstra

user_counts = [10, 30, 50, 70, 90, 150, 200]
num_satellites = 100
results_summary = []

# Step 1: Fixed satellite network
satellite_positions, connectivity = generate_synthetic_satellite_grid(
    num_satellites, lat_range=(30, 55), lon_range=(-140, 160)
)

for users_per_session in user_counts:
    print(f"\n\n=== Running for {users_per_session} users ===")

    # Step 2: Generate consistent user set
    sessions = generate_sessions(1, users_per_session)
    session = sessions[0]
    users = session.get_user()
    user_to_satellite = assign_users_to_closest_satellites(users, satellite_positions)

    # Step 3: Time and run Jacobi
    start_jacobi = time.perf_counter()
    res_jacobi = compute_satellite_routes_jacobi(users, user_to_satellite, satellite_positions, connectivity)
    end_jacobi = time.perf_counter()
    time_jacobi = end_jacobi - start_jacobi
    latencies_jacobi = [d['total_flow'] for d in res_jacobi.values() if d['flow']]

    # Step 4: Time and run Gauss-Seidel
    start_gs = time.perf_counter()
    res_gs = compute_satellite_routes_gauss_seidel(users, user_to_satellite, satellite_positions, connectivity)
    end_gs = time.perf_counter()
    time_gs = end_gs - start_gs
    latencies_gs = [d['total_flow'] for d in res_gs.values() if d['flow']]

    # Step 5: Time and run Dijkstra
    start_dij = time.perf_counter()
    res_dij = compute_satellite_routes_dijkstra(users, user_to_satellite, satellite_positions, connectivity)
    end_dij = time.perf_counter()
    time_dij = end_dij - start_dij
    latencies_dij = [d['latency'] for d in res_dij.values() if d['path']]

    results_summary.append({
        'users': users_per_session,
        'time_jacobi': time_jacobi,
        'time_gauss': time_gs,
        'time_dijkstra': time_dij,
        'avg_latency_jacobi': np.mean(latencies_jacobi) if latencies_jacobi else float('inf'),
        'avg_latency_gauss': np.mean(latencies_gs) if latencies_gs else float('inf'),
        'avg_latency_dijkstra': np.mean(latencies_dij) if latencies_dij else float('inf')
    })

# Plotting results
user_sizes = [r['users'] for r in results_summary]
time_jacobi = [r['time_jacobi'] for r in results_summary]
time_gauss = [r['time_gauss'] for r in results_summary]
time_dijkstra = [r['time_dijkstra'] for r in results_summary]

lat_jacobi = [r['avg_latency_jacobi'] for r in results_summary]
lat_gauss = [r['avg_latency_gauss'] for r in results_summary]
lat_dijkstra = [r['avg_latency_dijkstra'] for r in results_summary]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(user_sizes, time_jacobi, 'o-', label='Jacobi')
plt.plot(user_sizes, time_gauss, 's-', label='Gauss-Seidel')
plt.plot(user_sizes, time_dijkstra, '^-', label='Dijkstra')
plt.xlabel("Number of Users")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs User Count")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(user_sizes, lat_jacobi, 'o-', label='Jacobi')
plt.plot(user_sizes, lat_gauss, 's-', label='Gauss-Seidel')
plt.plot(user_sizes, lat_dijkstra, '^-', label='Dijkstra')
plt.xlabel("Number of Users")
plt.ylabel("Average Route Latency")
plt.title("Latency vs User Count")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
