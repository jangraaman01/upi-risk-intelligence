"""Indian city table + great-circle distance (used by the simulator, batch features and the streaming store)."""
import numpy as np

CITIES = {   # name: (lat, lon, population weight)
    "Mumbai": (19.076, 72.878, .15), "Delhi": (28.614, 77.209, .15), "Bengaluru": (12.972, 77.595, .12),
    "Hyderabad": (17.385, 78.487, .09), "Chennai": (13.083, 80.271, .08), "Kolkata": (22.573, 88.364, .07),
    "Pune": (18.520, 73.857, .07), "Ahmedabad": (23.023, 72.572, .06), "Jaipur": (26.912, 75.787, .05),
    "Lucknow": (26.847, 80.947, .04), "Kochi": (9.931, 76.267, .03), "Guwahati": (26.144, 91.736, .02),
}
CITY_NAMES = np.array(list(CITIES))
CITY_LAT = np.array([v[0] for v in CITIES.values()])
CITY_LON = np.array([v[1] for v in CITIES.values()])
CITY_P = np.array([v[2] for v in CITIES.values()]); CITY_P = CITY_P / CITY_P.sum()
CITY_INDEX = {n: i for i, n in enumerate(CITIES)}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def far_cities(min_km=600):
    """For every city index, the indices of cities at least `min_km` away (impossible-travel attacks)."""
    d = haversine_km(CITY_LAT[:, None], CITY_LON[:, None], CITY_LAT[None, :], CITY_LON[None, :])
    return [np.where(d[i] >= min_km)[0] for i in range(len(CITY_NAMES))]
