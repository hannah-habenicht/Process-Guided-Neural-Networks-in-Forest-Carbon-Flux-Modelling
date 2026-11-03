import random
import sys
import os
from itertools import product, combinations


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilis_shorts import save_json


# --- CONFIG ---
stations = [5, 12, 14, 21]
dict_year_station = {
    5:  [2001, 2002, 2004, 2005, 2006, 2007, 2008],
    12: [2004, 2005, 2008, 2009, 2010, 2011, 2012],
    14: [2001, 2003, 2004, 2005, 2006, 2007, 2008],
    21: [2002, 2003, 2004, 2005, 2006, 2007, 2008]
}

# --- USER-DEFINED LENGTHS ---
n_train_years = 3
n_val_years = 2
n_test_years = 2
max_inner_splits = 3
random.seed(42)

def get_split_hash(train, val, test):
    def flatten(cfg):
        return tuple(sorted((s, y) for s, ys in cfg.items() for y in ys))
    return hash((flatten(train), flatten(val), flatten(test)))

def fixed_train_val_combinations(years, n_train, n_val):
    """Return all (train, val) splits with fixed sizes."""
    splits = []
    for val_years in combinations(years, n_val):
        val_set = set(val_years)
        remaining = list(set(years) - val_set)
        if len(remaining) < n_train:
            continue
        for train_years in combinations(remaining, n_train):
            splits.append((sorted(train_years), sorted(val_years)))
    return splits

# --- MAIN LOOP ---
config_cv = {}
seen_hashes = set()

for outer_train_station in stations:
    for outer_test_station in stations:
        if outer_train_station == outer_test_station:
            continue

        train_years_pool = dict_year_station[outer_train_station]
        test_years_pool = dict_year_station[outer_test_station]

        if len(train_years_pool) < n_train_years + n_val_years:
            print(f"[SKIP] Station {outer_train_station} has insufficient training + validation years.")
            continue

        outer_key = f"{outer_train_station}_to_{outer_test_station}_train{n_train_years}_val{n_val_years}_test{n_test_years}"
        config_cv[outer_key] = {}

        inner_splits = fixed_train_val_combinations(train_years_pool, n_train_years, n_val_years)
        if len(inner_splits) > max_inner_splits:
            inner_splits = random.sample(inner_splits, max_inner_splits)

        for i, (train_yrs, val_yrs) in enumerate(inner_splits):
            # Use all test years if fewer than required
            if len(test_years_pool) <= n_test_years:
                test_yrs = sorted(test_years_pool)
            else:
                test_yrs = sorted(random.sample(test_years_pool, n_test_years))

            train_config = {
                'data::train::station_id': {str(outer_train_station): train_yrs}
            }
            val_config = {
                'data::val::station_id': {str(outer_train_station): val_yrs}
            }
            test_config = {
                'data::test::station_id': {str(outer_test_station): test_yrs}
            }

            split_hash = get_split_hash(
                train_config['data::train::station_id'],
                val_config['data::val::station_id'],
                test_config['data::test::station_id']
            )
            if split_hash in seen_hashes:
                continue
            seen_hashes.add(split_hash)

            inner_key = f"inner_{i}"
            config_cv[outer_key][inner_key] = [train_config, test_config, val_config]

            print(f"[{outer_key} | {inner_key}]")
            print(f"  Train: {train_yrs}")
            print(f"  Val  : {val_yrs}")
            print(f"  Test : {test_yrs}")


# --- SAVE ---
# save_json(config_cv, os.getcwd(), 'CV_train_one_station_predict_another')
