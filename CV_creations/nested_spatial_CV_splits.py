#test file 
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import combinations
from scipy.special import comb 
from utilis_shorts import save_json

# #station CV 
stations= [5,12,14,21] # skip 3 ? 
dict_year_station = { # 3 :[2001,2002,2004,2006, 2007, 2008],
                     5: [2001, 2002, 2004, 2005, 2006, 2007, 2008], #, 2009, 2010, 2011, 2012, 2013],
                     12:[2004, 2005, 2008, 2009, 2010, 2011, 2012],
                     14:[2001, 2003, 2004, 2005, 2006, 2007, 2008],
                     21:[2002, 2003, 2004, 2005, 2006, 2007, 2008]}

samples_inner = 2 # training years per station
test_station_inner = 1
training_station_inner = 2
test_split = 1 #number of test year
config= dict() #test dictionary
config_cv = dict()
config_cv['all_stations'] = dict_year_station
for i, stat in enumerate(stations): #outer loop left out station
    inner_station = stations.copy()
    inner_station.remove(stations[i])
    outer_station = stations[i]
    print(f'outer loop: {i+1}/{len(stations)} with outer station {outer_station} and inner {inner_station}')
    #inner_station_years ={i: random.sample(dict_year_station[i],samples_inner) for i in inner_station } #picks randomly number of samples 
    k = 0
    inner_station_years ={i: dict_year_station[i] for i in inner_station }
    #config_cv['outer_loop'] = {str(i): {'outer_station': outer_station, 'inner_station': inner_station}}

    for train_idx in combinations(inner_station, training_station_inner): #inner loop two trainins stations, one test/validation station
        k+=1
        train_stations = list(train_idx)
        test_station = list(set(inner_station)-set(train_idx))
        print(f'inner loop {k} \n')
        print(f'inner stations: train {train_stations} test{test_station}')
        #create configs
        train_config= config.copy()
        test_config = config.copy()
        val_config = config.copy()
        #create training and year of stations
        train_stat_year = {str(m): inner_station_years[m] for m in train_stations}
        #update training config
        train_config['data::train::station_id'] = train_stat_year

#        #split randomly between test and validation years
        pot_years = inner_station_years[test_station[0]] #works only for one test station
        test_years = random.sample(pot_years, test_split)
        val_years = list(set(pot_years)-set(test_years))
        
        #update test and validation config
        test_config['data::test::station_id'] ={str(k):  test_years for k in test_station} #has to be improved for more than one station
        val_config['data::val::station_id'] =  {str(k):  val_years  for k in test_station}
        cv_in = 'inner_loop_'+ str(k)
        cv_out = 'outer_station_' + str(outer_station)
        if k==1:
            config_cv[cv_out] ={cv_in: [train_config, test_config, val_config]}
        else:
            config_cv[cv_out].update({cv_in: [train_config, test_config, val_config]})
        print(f'train config {train_config}')
        print(f'test_config: {test_config}')
        print(f'val_config: {val_config} ')

print(config_cv)
path= os.getcwd()
#save_json(config_cv, path, 'CV_station_config')

# #CV time
# print('\n \n Temporal CV')
# #one station shuffle over years
# HP_train_years= 4 #make the decision, how many years are used for training
# outer_hold_out_years = 1

# inner_test_years = 1
# inner_val_years = 1

# station = 14

# years = dict_year_station[14]
# config = dict()
# k=0
# #rolling window approach
# for y in range(HP_train_years,len(years)): #holdout years in order of time to make sure the models also generalize well for future use 

#     inner_years =years[k:y]
#     hold_out_year =years[y] 
#     print(f'outer loop: with outer year {y} and inner {inner_years}')

#     #
#     k+=1
#     #inner HPO with remaing data
#     in_train_years = inner_years[:-2]
#     in_test_year = inner_years[-2]
#     in_val_year= inner_years[-1]

#     print(f'inner loop {k} with train years {in_train_years},test-years {in_test_year}, val_year {in_val_year}')
    
#     #update config
#     train_config= config.copy()
#     test_config = config.copy()
#     val_config = config.copy() 
    
#     train_config['data::train::station_id']={str(station):  in_train_years}
#     test_config['data::train::station_id'] ={str(station):  in_test_year} #has to be improved for more than one station
#     val_config['data::train::station_id'] =  {str(station):  in_val_year}

#     print(f'train config {train_config}')
#     print(f'test_config: {test_config}')
#     print(f'val_config: {val_config} ')

