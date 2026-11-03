#new dataset

import os
import torch
# import preles
from torch.utils.data import Dataset, DataLoader
from utilis_shorts import json_open

import numpy as np
import pandas as pd
from Data_preperation import ProfoundData
from Basic_dataset import Basic_dataset
from datetime import date
from typing import Any, Dict, Mapping, Optional, Union


class temporal_dataset(Basic_dataset):
    def __init__(self,path_dir:str, station_id:Dict,
                 x_para:list, y_para:list, 
                 input_window:int, output_window=1, 
                 label= 'present', normalise=None,sparse=False, previously_calibrated=True, device: bool=True, dataset=False,
                 calibrated_df=None,
                 modeltype= None, y_past_include = True,
                 normalization: Optional[Mapping[str, Any]] = None) -> None:
        
        '''
        Temporal Dataset: with optional sliding window and Label for 
        'future' or 'present' as output y (one timepoint ahead as last timpoint t of input or same ) 
        '''
        if not previously_calibrated:
            csv_file_name = f"preprocessed_data/"
            self.path_dir = os.path.join(path_dir, csv_file_name )

        if previously_calibrated:
            csv_file_name = f"previously_calibrated/"
            self.path_dir = os.path.join(path_dir, csv_file_name )

        super(temporal_dataset, self).__init__()
        
        #device
        self.test_device(device)
        self.in_wind= input_window
        self.out_wind = output_window
        self.label = label
        self.modeltype= modeltype
        self.dataset_type = 'temporal'
        self.previously_calibrated = previously_calibrated
        self.sparse = sparse

        self.x_parameter = x_para
        self.y_parameter = y_para

        # Get data
        #get data
        if calibrated_df is not None: # I am not sure, if it make sense, if there is already something in memory, spatial datset should not be called at all
            # We have a DataFrame in memory already
            # print("[DEBUG] Loading data from newly calibrated DataFrame (calibrated_df).")
            # NOTE: unlike spatial_dataset/Preles_data below, this branch never applies
            # `normalise` at all. Left as-is on purpose — it is currently unreachable
            # (load_CV_datasets' temporal branch is commented out), and silently
            # starting to normalise here would change any direct caller's data.
            self.data = calibrated_df
        else:
            self.get_all_data(self.path_dir,
                            station_id,
                            x_para, y_para,
                            normalise=normalise,
                            sparse=sparse,
                            dataset=dataset,
                            previously_calibrated=previously_calibrated,
                            normalization=normalization)


        #create temporal dataset
        self.create_temporal_dataset(y_past_include = y_past_include)

        if modeltype =='LSTM':
            self.data_x, self.data_y = self.rewrite_LSTM()


    def read_all_data(self):
        #restrict dataset, for huge dataset we should work differently e.g checkout Hp5
 
        x_list = list()
        y_list = list()

        x_list = self.data[self.X_t_parameter].to_numpy()
        y_list = self.data[self.Y_t_parameter].to_numpy()
        self.x = torch.tensor(x_list, dtype=torch.float32).to(self.device) # x_para + y_para [t-window_size:t]
        self.y = torch.tensor(y_list, dtype=torch.float32).to(self.device) # y_para t+1
        return self.x, self.y
    
    def __getitem__(self, index):
        
        if self.modeltype=='LSTM':
            x_i = torch.tensor(self.data_x[index], dtype=torch.float32).to(self.device)
            y_i = torch.tensor(self.data_y[index], dtype=torch.float32).to(self.device)
        else:
            data = self.data.iloc[index]
            x_list = data[self.X_t_parameter].values.tolist()
            y_list = data[self.Y_t_parameter].values.tolist()
            x_i = torch.tensor(x_list, dtype=torch.float32).to(self.device)
            y_i = torch.tensor(y_list, dtype=torch.float32).to(self.device)

        return x_i, y_i 
      
class spatial_dataset(Basic_dataset):
    def __init__(self,path_dir:str, station_id: Dict,
                 x_para:list, y_para:list, 
                 normalise=None, 
                 sparse_csv=False, 
                 previously_calibrated = True,
                 calibrated_df=None,
                 # make_new_predictions=False,
                 device:bool=True, dataset=False,
                 sparse=None,
                 normalization: Optional[Mapping[str, Any]] = None) -> None:
        super(spatial_dataset, self).__init__()
        if not previously_calibrated:
            csv_file_name = f"preprocessed_data/"
            self.path_dir = os.path.join(path_dir, csv_file_name )

        if previously_calibrated:
            csv_file_name = f"previously_calibrated/"
            self.path_dir = os.path.join(path_dir, csv_file_name )
        
        self.test_device(device)
        self.dataset_type = 'spatial'
        # self.make_new_predictions = make_new_predictions
        self.previously_calibrated = previously_calibrated
        self.sparse = sparse
        self.x_parameter = x_para
        self.y_parameter = y_para
        # self.normalise = normalise

        #get data
        if calibrated_df is not None: 
            # I am not sure, if it make sense, if there is already something in memory, spatial datset should not be called at all
            # I thought the 'spatial' dataset class was necessary to fulfill the pytorch dataloader requirements?
            # We should perhaps rename it to something else like default?
            
            # print("[DEBUG] Loading data from newly calibrated DataFrame (calibrated_df).")
            self.data = calibrated_df
            if normalise is not None:
                # print("[DEBUG] Normalizing in-memory calibrated DataFrame")
                # `normalization=None` self-fits on this frame. Correct only for the
                # train split — see load_CV_datasets' docstring.
                self.normalise(normalization=normalization)
        else:
            self.get_all_data(self.path_dir,
                            station_id, # startindex, stopindex,
                            x_para, y_para, normalise=normalise,
                            sparse=sparse,
                            sparse_csv=sparse_csv,
                            dataset=dataset,
                            previously_calibrated=previously_calibrated,
                            normalization=normalization)

    
    def read_all_data(self):
        #restrict dataset, for huge dataset we should work differently e.g checkout Hp5
        x_list = self.data[self.x_parameter].values.tolist()
        y_list = self.data[self.y_parameter].values.tolist()
               
        self.x = torch.tensor(x_list, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y_list, dtype=torch.float32).to(self.device)
        return self.x, self.y
    
    def __getitem__(self, index):
        data = self.data.iloc[index]
        x_list = data[self.x_parameter].values.tolist()
        y_list = data[self.y_parameter].values.tolist()
        x_i = torch.tensor(x_list, dtype=torch.float32).to(self.device)
        y_i = torch.tensor(y_list, dtype=torch.float32).to(self.device)

        return x_i, y_i
    
class Preles_data(Basic_dataset):
    '''
    calls Preles Dataset
    '''
    def __init__(self,path_dir:str, station_id:Dict,
                 x_para: list, y_para:list, y_phy_para: list,
                #  startindex: Union[date, int] = None, 
                #  stopindex: Union[date, int] = None, 
                #  required_preles_keys:dict= None, 
                #  calibrated:dict=None, 
                #  control=None,
                #  preintialised_output=None, 
                 device:bool=True,
                 # type_= '1_site_1_day',
                 dataset = True, 
                 calibrated_df=None, 
                 normalise=None,
                 sparse_csv=False,
                 previously_calibrated:bool = True, 
                 temporal:bool =False, window_input:Union[bool,int]=False,
                 window_output:Union[bool,int]=False,  modeltype= None,
                 error=False, y_past_include=True,
                 error_norm= False,
                 sparse=None,
                 normalization: Optional[Mapping[str, Any]] = None) -> None:
        '''

        '''
        self.previously_calibrated = previously_calibrated

        #self.test_device(device)
        if previously_calibrated:
            csv_file_name = "previously_calibrated/"
            self.path_dir = os.path.join(path_dir, csv_file_name)
        else:
            csv_file_name = "preprocessed_data/"
            self.path_dir = os.path.join(path_dir, csv_file_name ) #

        super(Preles_data,self).__init__()

        self.device = self.test_device(device)
        if self.device is None:
            #print(f" [DEBUG] Device found to be: {self.device}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print(f" [DEBUG] Device now set to: {self.device}")

        self.dataset = dataset
        self.temporal = temporal
        self.modeltype= modeltype
        self.error = error                                  # learning error of Preles model
        
        self.sparse = sparse                                # sparse datasets
        # self.path_dir = path_dir
        self.x_parameter = x_para
        self.y_parameter = y_para
        self.y_phy_para = y_phy_para

        #get data
        if calibrated_df is not None:
            # We have a DataFrame in memory already
            # print("Loading data from newly calibrated DataFrame (calibrated_df).")
            self.data = calibrated_df
            if normalise is not None:
                # print("[DEBUG] Normalizing in-memory calibrated DataFrame")
                # `normalization=None` self-fits on this frame. Correct only for the
                # train split — see load_CV_datasets' docstring.
                self.normalise(normalization=normalization)

        else:
            self.get_all_data(self.path_dir, station_id=station_id,
                             x_parameters=x_para,  y_parameters=y_para,
                             y_phys_para= y_phy_para,
                            dataset= dataset,
                            normalise=normalise,
                            sparse=sparse,
                            previously_calibrated=previously_calibrated,
                            error=error,
                            error_norm=error_norm,
                            sparse_csv=sparse_csv,
                            normalization=normalization,
                            )

        if self.temporal == 'temporal':
            self.in_wind= window_input
            self.out_wind = window_output
            self.create_temporal_dataset(physics=True, error=error, y_past_include= y_past_include)
            if  self.modeltype =='LSTM' and error== False: #todo for error prediction
                self.data_x, self.data_y, self.data_p = self.rewrite_LSTM(physics=True)
            elif self.modeltype =='LSTM' and error== True:
                self.data_x, self.data_err_x, self.data_y, self.data_p, self.data_err_y = self.rewrite_LSTM(physics=True, error=True)
        
    def get_pd_data(self):
        return self.data
    
    def __getitem__(self, index):
        data = self.data.iloc[index]
        # print(f"\n [DEBUG] Inside __getitem__(): Moving data to device: {self.device}")
        
        if self.temporal ==  'temporal':
            if  self.modeltype =='LSTM' and self.error ==False:
                x_list = self.data_x[index]
                y_list = self.data_y[index]
                y_pi_list =  self.data_p[index]

            elif self.modeltype =='LSTM' and self.error == True:
                x_list = self.data_x[index]
                x_err_list = self.data_err_x[index]
                y_list = self.data_y[index]
                y_err_list = self.data_err_y[index]
                y_pi_list =  self.data_p[index]
            else:
                x_list = data[self.X_t_parameter].values.tolist()
                y_list = data[self.Y_t_parameter].values.tolist()
                y_pi_list = data[self.Y_t_physics].values.tolist()

                if self.error and self.modeltype !='LSTM':
                    x_err_list = data[self.err_t_x].values.tolist()
                    y_err_list = data[self.err_t_y].values.tolist()

        elif self.temporal !=  'temporal' and self.error :
                x_err_list = data[self.err_para].values.tolist()
                y_err_list = data[self.err_para].values.tolist()
                x_list = data[self.x_parameter].values.tolist()
                y_list = data[self.y_parameter].values.tolist()
                y_pi_list = data[self.y_phy_para].values.tolist()
        else:
            x_list = data[self.x_parameter].values.tolist()
            y_list = data[self.y_parameter].values.tolist()
            y_pi_list = data[self.y_phy_para].values.tolist()

        # if not self.dataset:  # Use self.dataset to refer to the instance variable
        #         y_pi_list = self.preles_calling(index)
        #         #y_pi_list = data[self.y_phy_para].values.tolist()

        # Convert lists to tensors
        x_i = torch.tensor(x_list, dtype=torch.float32).to(self.device)
        y_i = torch.tensor(y_list, dtype=torch.float32).to(self.device)    
        y_pi = torch.tensor(y_pi_list, dtype=torch.float32).to(self.device)

        if self.error :
            y_err = torch.tensor(y_err_list, dtype=torch.float32).to(self.device)
            x_err = torch.tensor( x_err_list, dtype=torch.float32).to(self.device)
            y_pi = (y_pi,y_err)
            x_i = (x_i,x_err)

        return x_i, y_i ,y_pi 
    
    def read_all_data(self):
        #restrict dataset, for huge dataset we should work differently e.g checkout Hp5
        x_list = list()
        y_list = list()
        y_p_list = list()

        if self.temporal ==  'temporal':
            if  self.modeltype =='LSTM' and self.error ==False:
                x_list = self.data_x
                y_list = self.data_y
                y_p_list =  self.data_p
            elif self.modeltype =='LSTM' and self.error == True:
                x_list = self.data_x
                x_err_list = self.data_err_x
                y_list = self.data_y
                y_err_list = self.data_err_y
                y_p_list =  self.data_p
            else:
                x_list = self.data[self.X_t_parameter].values.tolist()
                y_list = self.data[self.Y_t_parameter].values.tolist()
                y_p_list = self.data[self.Y_t_physics].values.tolist()

                if self.error and self.modeltype !='LSTM': #for error model
                    x_err_list = self.data[self.err_t_x].values.tolist()
                    y_err_list = self.data[self.err_t_y].values.tolist()

        elif self.temporal !=  'temporal' and self.error :
            x_err_list = self.data[self.err_para].values.tolist()
            y_err_list = self.data[self.err_para].values.tolist()
            x_list = self.data[self.x_parameter].values.tolist()
            y_list = self.data[self.y_parameter].values.tolist()
            y_p_list = self.data[self.y_phy_para].values.tolist()
        else: 
            x_list = self.data[self.x_parameter].values.tolist()
            y_list = self.data[self.y_parameter].values.tolist()
            y_p_list = self.data[self.y_phy_para].values.tolist()
       
        self.x = torch.tensor(x_list, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y_list, dtype=torch.float32).to(self.device)
        self.y_p = torch.tensor(y_p_list, dtype=torch.float32).to(self.device)

        if self.error:
            y_err = torch.tensor(y_err_list, dtype=torch.float32).to(self.device)
            x_err = torch.tensor( x_err_list, dtype=torch.float32).to(self.device)
            self.y_p = (self.y_p,y_err)
            self.x = (self.x,x_err)

        return self.x, self.y, self.y_p
    

if __name__=='__main__':
    ## from file_name import build_config                   ## change the file name to change the configuration file 
    from Config_Files.config_spatial import build_config    ## explicitly set which configuration file you wish to use

    config = build_config()
    save = config["cfg::save"]

    data = spatial_dataset(config['data::train::path_dir'], 
                    station_id=config['data::train::station_id'],
                    x_para=config['data::train::x_parameter'], 
                    y_para=config['data::train::y_parameter'],
                    normalise=config['data::normalise'],
                    sparse=None,
                    previously_calibrated=False,
                    dataset=True
                        ) 
    
    # data = temporal_dataset(config['data::train::path_dir'], 
    #                 station_id=config['data::train::station_id'],
    #                 startindex=config['data::train::startdate'], 
    #                 stopindex=config['data::train::stopdate'], 
    #                 x_para=config['data::train::x_parameter'], 
    #                 y_para=config['data::train::y_parameter'],
    #                 normalise=config['data::normalise'],
    #                 modeltype=config['model::modeltype'],
    #                 sparse=True,
    #                 calibration=True,
    #                 dataset=True, 
    #                 input_window=1,
    #                 output_window=1
    #             )
    
    # data.read_all_data()
    data.__getitem__(1)
                         
    
    # data = temporal_dataset(config['data::train::path_dir'], 
    #                 station_id=config['data::train::station_id'],
    #                 startindex=config['data::train::startdate'], 
    #                 stopindex=config['data::train::stopdate'], 
    #                 x_para=config['data::train::x_parameter'], 
    #                 y_para=config['data::train::y_parameter'],
    #                 normalise=config['data::normalise'],
    #                 dataset=True, input_window=3, output_window=2)
    

    # # Now you have train_data and test_data as DataFrames with predictions
    # print(data.head())
    # print(test_data.head())

    # test_data.rewrite_LSTM()
    # test_data.read_all_data()
    # test_data.__getitem__(1)

    #train_dataloader = DataLoader(train_data, batch_size=10)
    
    print('done')
