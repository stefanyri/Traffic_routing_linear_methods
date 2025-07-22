import heapq
import numpy as np

class USER:
    def __init__(self, user_id, city, latitude, longitude, create_time, session_id):
        self.user_id = user_id
        self.city = city
        self.latitude = latitude
        self.longitude = longitude
        self.create_time = create_time
        self.session_id = session_id

    def get_id(self):
        return self.user_id

    def get_city(self):
        return self.city

    def get_location(self):
        return self.latitude, self.longitude


def find_user_center(user_list):
    """
    Compute average (lat, lon) for a list of users.
    """
    latitudes = [u.latitude for u in user_list]
    longitudes = [u.longitude for u in user_list]
    return float(np.mean(latitudes)), float(np.mean(longitudes))


def k_center(user_list, satellite_positions, k=3):
    """
    Find k closest satellites to the user cluster center.

    Args:
        user_list (list of USER)
        satellite_positions (np.ndarray): shape [N, 2]
        k (int): number of satellites to return

    Returns:
        list of satellite indices (int)
    """
    center_lat, center_lon = find_user_center(user_list)
    distances = np.sum((satellite_positions - np.array([center_lat, center_lon])) ** 2, axis=1)
    closest_ids = heapq.nsmallest(k, range(len(distances)), key=lambda idx: distances[idx])
    return closest_ids
