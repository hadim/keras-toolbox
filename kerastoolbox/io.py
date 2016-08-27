import h5py


def load_data(dataset_fname):
    """Load X and Y arrays from disk
    """
    with h5py.File(dataset_fname, "r") as f:
        X = f["X"][()]
        Y = f["Y"][()]
    return X, Y


def save_data(dataset_fname, X, Y):
    """Load X and Y arrays from disk
    """
    with h5py.File(dataset_fname, "w") as f:
        f.create_dataset("X", data=X, dtype=X.dtype)
        f.create_dataset("Y", data=Y, dtype=Y.dtype)
        
