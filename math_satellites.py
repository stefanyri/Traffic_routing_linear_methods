import numpy as np
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 299792.458  # km/s


def generate_synthetic_satellite_grid(num_satellites=100, lat_range = (30, 55),
    lon_range = (-140, 160)):
    """
    Generate satellite grid over a custom region matching user city spread.
    Args:
        num_satellites: total satellites
        lat_range: tuple (min_lat, max_lat)
        lon_range: tuple (min_lon, max_lon)
    Returns:
        satellite_positions: np.array shape [N, 2] (lat, lon)
        connectivity_dict: {sat_id: [neighbor_ids]}
    """
    num_orbits = int(np.sqrt(num_satellites))
    sats_per_orbit = (num_satellites + num_orbits - 1) // num_orbits

    lats = np.linspace(lat_range[0], lat_range[1], sats_per_orbit)
    lons = np.linspace(lon_range[0], lon_range[1], num_orbits, endpoint=False)

    satellite_positions = []
    sat_id_map = {}      # (i, j) → sat_id
    reverse_map = {}     # sat_id → (i, j)

    sat_id = 0
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            if sat_id >= num_satellites:
                break  # don’t exceed
            satellite_positions.append((lat, lon))
            sat_id_map[(i, j)] = sat_id
            reverse_map[sat_id] = (i, j)
            sat_id += 1

    satellite_positions = np.array(satellite_positions)

    connectivity = {}
    for sid in range(num_satellites):
        if sid not in reverse_map:
            continue
        i, j = reverse_map[sid]
        neighbors = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            neighbor_key = (ni % num_orbits, nj)
            if neighbor_key in sat_id_map:
                neighbors.append(sat_id_map[neighbor_key])

        connectivity[sid] = neighbors

    return satellite_positions, connectivity

def plot_satellite_grid(satellite_positions):
    plt.figure(figsize=(12, 6))
    lats, lons = satellite_positions[:, 0], satellite_positions[:, 1]
    plt.scatter(lons, lats, c='red', s=30)
    for i, (lat, lon) in enumerate(satellite_positions):
        plt.text(lon, lat, f"S{i}", fontsize=6, color='black', ha='center', va='center')
    plt.title("Synthetic Satellite Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def calculate_latency(user_lat, user_lon, sat_lat, sat_lon, base_latency_ms=50):
    """Calculate latency in milliseconds between user and satellite."""
    distance_km = np.sqrt((sat_lat - user_lat) ** 2 + (sat_lon - user_lon) ** 2) * 111
    latency_ms = (distance_km / SPEED_OF_LIGHT) * 1000
    return base_latency_ms + latency_ms


if __name__ == "__main__":
    num_satellites = 100
    lat_range = (30, 55)
    lon_range = (-140, 160)
    sats, connectivity = generate_synthetic_satellite_grid(num_satellites,lat_range,lon_range)
    plot_satellite_grid(sats)

    print("Satellite Connectivity (immediate neighbors):")
    for sid in sorted(connectivity):
        neighbors = ", ".join(f"S{n}" for n in connectivity[sid])
        print(f"S{sid} connects to: {neighbors}")
