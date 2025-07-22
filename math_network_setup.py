import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from math_users_information import USER
from math_session_information import SESSION
from math_satellites import generate_synthetic_satellite_grid, calculate_latency


CITY_COORDINATES = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "Paris": (48.8566, 2.3522),
    "Sydney": (33.8688, 151.2093),
}


def generate_sessions(num_sessions, max_users_per_session):
    sessions = []
    user_id = 0

    for session_id in range(num_sessions):
        session = SESSION(session_id)
        for _ in range(max_users_per_session):
            city = random.choice(list(CITY_COORDINATES.keys()))
            base_lat, base_lon = CITY_COORDINATES[city]
            lat = base_lat + random.uniform(-2.0, 2.0)
            lon = base_lon + random.uniform(-2.0, 2.0)
            timestamp = random.randint(1609459200, 1672444800)

            user = USER(user_id, city, lat, lon, timestamp, session_id)
            session.add_user(user)
            user_id += 1

        sessions.append(session)
    return sessions


def assign_users_to_closest_satellites(users, satellite_positions):
    user_to_satellite = {}

    for user in users:
        user_lat, user_lon = user.get_location()
        distances = np.sqrt((satellite_positions[:, 0] - user_lat) ** 2 +
                            (satellite_positions[:, 1] - user_lon) ** 2)
        closest_idx = np.argmin(distances)
        user_to_satellite[user.get_id()] = closest_idx

    return user_to_satellite


def plot_user_satellite_graph(users, satellite_positions, user_to_satellite_map):
    fig, ax = plt.subplots(figsize=(12, 6))

    sat_lats = satellite_positions[:, 0]
    sat_lons = satellite_positions[:, 1]
    ax.scatter(sat_lons, sat_lats, c='red', label='Satellites')

    user_lats = [u.get_location()[0] for u in users]
    user_lons = [u.get_location()[1] for u in users]
    ax.scatter(user_lons, user_lats, c='blue', label='Users')

    for i, (lat, lon) in enumerate(satellite_positions):
        ax.text(lon, lat, f"S{i}", fontsize=7, color='black', ha='center', va='center')

    for user in users:
        user_lat, user_lon = user.get_location()
        ax.text(user_lon, user_lat, f"U{user.get_id()}", fontsize=7, color='blue')

        sat_id = user_to_satellite_map[user.get_id()]
        sat_lat, sat_lon = satellite_positions[sat_id]
        ax.plot([user_lon, sat_lon], [user_lat, sat_lat], color='green', lw=0.5)

    #ax.set_xlim(-130, 170)
    #ax.set_ylim(0, 70)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("User ↔ Satellite Connections")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_colored_user_satellite_graph(users, satellite_positions, user_to_satellite_map):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Satellite positions
    sat_lats = satellite_positions[:, 0]
    sat_lons = satellite_positions[:, 1]
    ax.scatter(sat_lons, sat_lats, color='red', label='Satellites', alpha=0.7)
    for sat_id, (lat, lon) in enumerate(satellite_positions):
        ax.text(lon, lat, f"S{sat_id}", color='black', fontsize=6, ha='center', va='center')

    # User positions by city
    city_colors = {
        "New York": "green",
        "Los Angeles": "orange",
        "London": "blue",
        "Tokyo": "purple",
        "Paris": "pink",
        "Sydney": "cyan",
    }

    for user in users:
        user_lat, user_lon = user.get_location()
        city = user.get_city()
        color = city_colors.get(city, "gray")
        ax.scatter(user_lon, user_lat, color=color, alpha=0.7, s=30, label=city)
        ax.text(user_lon, user_lat, f"U{user.get_id()}", fontsize=6, color='black', ha='center', va='center')

        # User → Satellite line
        sat_id = user_to_satellite_map[user.get_id()]
        sat_lat, sat_lon = satellite_positions[sat_id]
        ax.plot([user_lon, sat_lon], [user_lat, sat_lat], color='gray', lw=0.5, alpha=0.5)

    # Remove duplicate city entries in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')


    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('User ↔ Satellite Connections (Colored by City)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def print_user_satellite_pairs(users, user_to_satellite):
    print("\nUser-Satellite Assignment:")
    for user in users:
        print(f"User {user.get_id()} → Satellite {user_to_satellite[user.get_id()]}")


def print_satellite_connectivity(connectivity_dict):
    print("\nSatellite Connectivity (Immediate Neighbors):")
    for sat_id in sorted(connectivity_dict):
        neighbors = ", ".join(f"S{n}" for n in connectivity_dict[sat_id])
        print(f"S{sat_id} connects to: {neighbors}")


if __name__ == "__main__":
    num_satellites = 100
    num_sessions = 1
    users_per_session = 20

    # Step 1: Create satellite grid
    satellite_positions, connectivity = generate_synthetic_satellite_grid(
        num_satellites,
        lat_range=(30, 55),
        lon_range=(-140, 160)
    )

    # Step 2: Generate users
    sessions = generate_sessions(num_sessions, users_per_session)
    session = sessions[0]
    users = session.get_user()

    # Step 3: Assign users to satellites
    user_to_satellite_map = assign_users_to_closest_satellites(users, satellite_positions)

    print_user_satellite_pairs(users, user_to_satellite_map)
    print_satellite_connectivity(connectivity)

    # Step 4: Plot and print
    #plot_user_satellite_graph(users, satellite_positions, user_to_satellite_map)
    plot_colored_user_satellite_graph(session.get_user(), satellite_positions, user_to_satellite_map)


