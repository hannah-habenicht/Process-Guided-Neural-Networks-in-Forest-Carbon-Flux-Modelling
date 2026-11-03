#nested Cv for 
import os
from utilis_shorts import save_json

print('new temporal CV splits')

#avaible years for station
station=14
dict_year_station = {14:[2001, 2003, 2004, 2005, 2006, 2007, 2008]}

#inner loop 
train_years_len = 2 #5
inn_test_year = 1

#define sequentiell hold out years for the outer loop
hold_out_len = 2
years = dict_year_station[station]
avaibl_years = len(dict_year_station[station])

in_year= avaibl_years-hold_out_len

#how many inner and outer loops do we want to have as max
inner_loop_times_max = 3
outer_loop_times_max=3

#outer loop we start not at the first years, but we keep enough distance so we have squential data for training before
config_cv = dict()
config_cv['station'] = station
config_cv['all_years'] = list(dict_year_station.values())

out_l = 1
for k in range(1, avaibl_years, hold_out_len):
    if k + hold_out_len > avaibl_years:
        break
    hold_out_years = years[k:k+hold_out_len]
    if k==0:
        out_train_years = years[-train_years_len:] #use last years
    elif k < train_years_len:
        out_train_years =years[:k] + years[-train_years_len+len(years[:k]):]
    else:
        out_train_years =years[k-train_years_len:k]
        
    #inner years
    inner_years = years.copy()
    del inner_years[k:k+hold_out_len]
    
    print(f'Outer loop {out_l}')
    print(f'hold out years: {hold_out_years}')
    print(f'train_outer: {out_train_years}')
    print(f'Inner years {inner_years}')
  
    config_cv[f'out_l_{out_l}']={'hold_out_years': hold_out_years, 'train_hold_out':out_train_years, 'inner_years': inner_years}
    in_l=0
    for i in range(0, len(inner_years), inn_test_year):
        in_l+=1
        # print(f'steps:i {i}')
        if in_l > inner_loop_times_max:
            print('max inner loop reached')
            break
        test_years = inner_years[i:i+inn_test_year]
        if i==0:
            config_cv[f'out_l_{out_l}']['inner_loops']= {}
            val_years = inner_years[i+inn_test_year:i+2*inn_test_year]
            train_years = inner_years[-train_years_len:] #use last years
        elif  i<train_years_len:
            train_years =inner_years[:i]
            train_years.extend(inner_years[-train_years_len+len(train_years):])
            val_years=inner_years[i+inn_test_year:i+2*inn_test_year]
        elif i>= train_years_len and i < len(inner_years)- 2*inn_test_year:
            train_years =inner_years[i-train_years_len:i]
            val_years = inner_years[i+inn_test_year:i+2*inn_test_year]
            #train_years.extend(inner_years[-train_years_len+i:])
            
        elif i>= len(inner_years)-2*inn_test_year:
            if i < len(inner_years)-inn_test_year:
                val_years= inner_years[i+inn_test_year:]
                val_years.extend(inner_years[:inn_test_year-len(val_years)])
                train_years = inner_years[i-train_years_len:i]
            else:
                val_years= inner_years[:inn_test_year]
                train_years = inner_years[i-train_years_len:i]
        # else:
        #     train_years =inner_years[i-train_years_len:i]
        
        config_cv[f'out_l_{out_l}']['inner_loops'].update({f'inner_loop{in_l}':{'train':train_years,'test':test_years, 'val':val_years}})
        print(f"    Inner loop {in_l}: train: {train_years}  test: {test_years}  val: {val_years};  Steps:{i} ")
       
    out_l+=1
save= os.getcwd()
name = f'new_temporal_cv_train_len{train_years_len}_station_{station}'
save_json(config_cv,path=save, name=name)


