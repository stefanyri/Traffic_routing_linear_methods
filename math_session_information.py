import numpy as np
from math_users_information import k_center

class SESSION:
    def __init__(self, session_id):
        self.id = session_id
        self.user = []
        self.user_satellite_map = {}  # Map user_id -> satellite_id

    def get_user(self):
        return self.user

    def add_user(self, to_add):
        self.user.append(to_add)

    def assign_k_center_satellites(self, satellite_positions, k):
        """
        Assign each user to their nearest among k satellites
        closest to the user group center.
        Args:
            satellite_positions (np.ndarray): shape [N, 2]
            k (int): number of candidate satellites (from k-center)
        """
        self.user_satellite_map = {}
        k_satellite_indices = k_center(self.user, satellite_positions, k)
        k_sat_positions = satellite_positions[k_satellite_indices]

        for user in self.user:
            user_lat, user_lon = user.get_location()
            distances = np.sqrt((k_sat_positions[:, 0] - user_lat) ** 2 +
                                (k_sat_positions[:, 1] - user_lon) ** 2)
            closest_idx = np.argmin(distances)
            closest_sat_id = k_satellite_indices[closest_idx]
            self.user_satellite_map[user.get_id()] = closest_sat_id
