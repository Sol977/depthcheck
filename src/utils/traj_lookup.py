
import numpy as np

def lookup_Twc(traj_map: dict, ts_str: str, tol: float = 1e-3):
    """
    Robust lookup for Twc by timestamp string.
    - First try exact key
    - Else nearest float key within tolerance (seconds)
    """
    if ts_str in traj_map:
        return traj_map[ts_str]
    try:
        ts = float(ts_str)
    except Exception:
        return None

    # build numeric index once per traj_map instance
    cache = getattr(lookup_Twc, "_cache", None)
    if cache is None or cache.get("id") != id(traj_map):
        keys_f = []
        vals = []
        for k, v in traj_map.items():
            if not isinstance(k, str):
                continue
            try:
                keys_f.append(float(k))
                vals.append(v)
            except Exception:
                continue
        if not keys_f:
            lookup_Twc._cache = {"id": id(traj_map), "keys": np.array([], dtype=np.float64), "vals": []}
        else:
            keys = np.array(keys_f, dtype=np.float64)
            order = np.argsort(keys)
            lookup_Twc._cache = {"id": id(traj_map), "keys": keys[order], "vals": [vals[i] for i in order]}

    keys = lookup_Twc._cache["keys"]
    vals = lookup_Twc._cache["vals"]
    if keys.size == 0:
        return None

    j = int(np.searchsorted(keys, ts))
    cand = []
    for jj in (j-1, j, j+1):
        if 0 <= jj < keys.size:
            cand.append((abs(float(keys[jj] - ts)), vals[jj]))
    if not cand:
        return None
    d, Twc = min(cand, key=lambda x: x[0])
    return Twc if d <= tol else None
