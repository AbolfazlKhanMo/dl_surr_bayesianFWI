import numpy as np
from pathlib import Path

def load_data(path, n_values=None, flatten_order="C"):
    """
    Load data.bin into a NumPy array.
    
    Parameters
    ----------
    path : str or Path
        Path to the .bin file (e.g., target_data.bin).
    n_values : int, optional
        Expected number of float32 values. If given, will validate.
    flatten_order : {"C", "F"}, optional
        Order in which the data was originally flattened (must match save).
    
    Returns
    -------
    data : np.ndarray
        1D NumPy array of float32 values.
    """
    path = Path(path)
    data = np.fromfile(path, dtype=np.float32)

    if n_values is not None and data.size != n_values:
        raise ValueError(f"Expected {n_values} values, found {data.size}")

    return data
