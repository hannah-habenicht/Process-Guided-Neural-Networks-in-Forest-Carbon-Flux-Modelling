#test file 
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import combinations
from scipy.special import comb 
from utilis_shorts import save_json

# #station CV 
stations= ['IT-Col','FI-Hyy','FR-LBr','DK-Sor'] 
dict_year_station = { 'IT-Col': [2007, 2008, 2009, 2010, 2011, 2012, 2013],
                      'FI-Hyy': [2004, 2005, 2008, 2009, 2010, 2011, 2012],
                      'FR-LBr': [2001, 2002, 2004, 2005, 2006, 2007, 2008],
                      'DK-Sor': [2002, 2003, 2004, 2005, 2006, 2007, 2008]}

samples_inner = 4   #number of training years per station
test_split = 2 
val_splits =2     #number of test years

training_station_inner = 2 # how many stations to use for inner training
test_station_inner = 2

random_seed = 42
random.seed(random_seed)

def sample_temporal_years(years, n_samples, seed=None):
    """Try to select n_samples consecutive years (allowing for end wraparound if needed)."""
    rng = random.Random(seed)
    sorted_years = sorted(years)
    max_start = len(sorted_years) - n_samples

    if max_start <= 0:
        return sorted_years  # not enough years, return all

    # Try multiple times to find a temporally close block
    for _ in range(10):
        idx =rng.randint(0, len(sorted_years) - 1)
        sample = sorted_years[idx:idx+n_samples]

        if len(sample) == n_samples:
            return sample

    # Fallback: random with wraparound allowed
    idx = rng.randint(0, len(sorted_years) - 1)
    return (sorted_years * 2)[idx:idx + n_samples]

def get_training_hash(train_config):
    station_years = train_config['data::train::station_id']
    flat_list = [(s, y) for s, years in station_years.items() for y in sorted(years)]
    return hash(tuple(sorted(flat_list)))

config= dict() #test dictionary
config_cv = dict()
config_cv['all_stations'] = dict_year_station # Years available for each station

for i, stat in enumerate(stations): #outer loop left out station
    random.seed(random_seed)  # Ensure reproducibility for each outer loop
    inner_station = stations.copy()
    inner_station.remove(stations[i])
    outer_station = stations[i]
    cv_out = 'outer_station_' + str(outer_station)
    print(f'outer loop: {i+1}/{len(stations)} with outer station {outer_station} and inner {inner_station} \n')
    # inner_station_years ={i: random.sample(dict_year_station[i],samples_inner) for i in inner_station } #picks randomly number of samples 
   
    outer_train_years = {
        s: sample_temporal_years(
            dict_year_station[s],
            samples_inner,
            seed=random_seed + i * 100 + inner_idx,
        )
        for inner_idx, s in enumerate(inner_station)
    }
    outer_test_years = dict_year_station[outer_station][-3:]

    # ----- PRINT OUTER LOOP YEARS -----
    print(f'Outer loop TRAIN years per station:')
    for s in inner_station:
        print(f' Station {s}: {outer_train_years[s]}')
    print(f'Outer loop TEST years for station {outer_station}: {outer_test_years}')
    
    config_cv[cv_out] = dict()
    config_cv[cv_out]['outer_split'] = {
        'train': outer_train_years,
        'test': {outer_station: outer_test_years}
    }
    
    k = 0
    for train_idx in combinations(inner_station, training_station_inner): #inner loop two trainins stations, one test/validation station
        k+=1
        train_stations = list(train_idx)
        test_station = list(set(inner_station)-set(train_idx))
        print(f'  Inner loop {k}')
        print(f'  Inner stations: train {train_stations} test{test_station}')
        train_station_years = {
            station: sample_temporal_years(dict_year_station[station], samples_inner)
            for station in train_stations
        }
        #create configs
        train_config= config.copy()
        test_config = config.copy()
        val_config = config.copy()
        #create training and year of stations
        train_stat_year = train_station_years #{str(m): inner_station_years[m] for m in train_stations}
        #update training config
        train_config['data::train::station_id'] = train_stat_year

#       # split randomly between test and validation years
        pot_years = dict_year_station[test_station[0]] #works only for one test station
        test_years = random.sample(pot_years, test_split)
        val_years = random.sample(list(set(pot_years)-set(test_years)), val_splits)
        
        #update test and validation config
        test_config['data::test::station_id'] = {station: test_years for station in test_station} #has to be improved for more than one station
        val_config['data::val::station_id'] = {station: val_years for station in test_station}

        # to do: outerloop, set specific years for outer station
        cv_in = 'inner_loop_'+ str(k)
        cv_out = 'outer_station_' + str(outer_station)
        
        config_cv[cv_out][cv_in] = [train_config, test_config, val_config]
        # if k==1:
        #     config_cv[cv_out] ={cv_in: [train_config, test_config, val_config]}
        # else:
        #     config_cv[cv_out].update({cv_in: [train_config, test_config, val_config]})
        
        print(f'  train config {train_config}')
        print(f'  test_config: {test_config}')
        print(f'  val_config: {val_config}\n ')

# seen_hashes = set()
# train_hash = get_training_hash(train_config)
# assert train_hash not in seen_hashes, "Duplicate training set detected!"
# seen_hashes.add(train_hash)

print(f'\n', config_cv)
path= "configs/CV_splits"
#save_json(config_cv, path, f'CV_station_len{3 * samples_inner}_config')

