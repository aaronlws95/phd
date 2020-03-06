import lmdb
import pickle
import numpy    as np
from tqdm       import tqdm

def get_keys(keys_cache_file):
    """ Get keys for LMDB """
    return pickle.load(open(keys_cache_file, "rb"))

def get_env(dataroot):
    """ Get LMDB environment """
    return lmdb.open(dataroot, readonly=True, lock=False,
                     readahead=False, meminit=False)

def get_data_from_txn(txn, key, dtype, reshape_dim):
    """ Get data from LMDB transaction """
    buf = txn.get(key.encode("ascii"))
    data_flat = np.frombuffer(buf, dtype=dtype)
    return data_flat.reshape(reshape_dim)

def read_lmdb_env(key, env, dtype, reshape_dim):
    """ Get data from LMDB environment """
    with env.begin(write=False) as txn:
        return get_data_from_txn(txn, key, dtype, reshape_dim)

def read_lmdb_dataroot(key, dataroot, dtype, reshape_dim):
    """ Get data from LMDB dataroot """
    env = get_env(dataroot)
    return read_lmdb_env(key, env, dtype, reshape_dim)

def read_all_lmdb_dataroot(keys, dataroot, dtype, reshape_dim):
    """ Read all data into Numpy array from LMDB dataroot """
    env = get_env(dataroot)
    all_data = []
    with env.begin(write=False) as txn:
        for key in tqdm(keys):
            all_data.append(get_data_from_txn(txn, key, dtype, reshape_dim))
    return np.asarray(all_data)

def write_to_lmdb(keys, dataset, save_dir):
    """ Write data to LMDB """
    data_size_per_data = dataset[keys[0]].nbytes
    data_size = data_size_per_data * len(dataset)
    env = lmdb.open(save_dir, map_size=data_size * 10)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for key in tqdm(keys):
            key_byte = key.encode("ascii")
            data = dataset[key]
            txn.put(key_byte, data)