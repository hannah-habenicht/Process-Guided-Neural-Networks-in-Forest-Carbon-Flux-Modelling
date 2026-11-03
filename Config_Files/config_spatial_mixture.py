# config_spatial.py

import json
import torch
import os
from utilis_shorts import save_json, json_open
from dotenv import load_dotenv

load_dotenv()

def build_config():
    cfg= {}
    #mixture set up
    cfg['num_trainable_experts']=2
    cfg['hard_expert_id']=1 #index where preles is
    cfg['top_k']=2
    cfg["warmup_steps"]=2000
    cfg["ramp_steps"]= 2000
    cfg["noise_steps"]=3000
    cfg["alpha_entropy"]=0.001

    # Model architecture
    cfg['model::modeltype'] =  'MLP' #MLP' #'Adapted_error' # 'MLP' #'LSTM 'Adapted_error'
    # For MLP
    cfg['model::layer_number']= 4 #MLP Layer number
    cfg['model::input_size']= 5
    cfg['model::output_size']= 2
    cfg['model::hidden_size'] =  [5,10,28]
    cfg['model::activation'] =  ['Relu','Relu','Relu',None] #* cfg['model::layer_number']  #'Tanh', None
    cfg['model::dropout_list'] = 0.2 #* cfg['model::layer_number']
    cfg['model::bias_list'] = [True] * cfg['model::layer_number'] #[True, True, True]
    cfg['model::save_path'] = None
    cfg['model::seed'] = 42 

    # Model training
    cfg['training::cuda'] = True
    cfg['training::learning_rate']= 3e-4
    cfg['training::momentum']= 0 #SGD
    cfg['training::weight_decay'] = 0
    cfg['training::dampening']= 0 #SGD
    cfg['training::beta_1']= 0.9 #Adam
    cfg['training::beta_2']= 0.999 #Adam
    cfg['training::optimizer'] = 'AdamW' #AdamW, SGD ...
    cfg['training::loss_type'] = 'MSE' #'MAE' #MSE' #MA
    cfg['training::physical_loss_regul_term']= 0.2
    cfg['training::epochs'] = 100
    cfg['training::seed'] = 42
    cfg['training::acc_metric'] = 'MAE' #'R2','MSE'#
    cfg['training::batchsize'] = 16
    cfg['data::test::batchsize'] = 16

    # Experimental Version 
    cfg['model::prediction_type']= 'spatial'        #temporal' #'spatial'
    cfg['training::loss_physic_type'] = 'parallel'     #'naive' # parallel, regularized, #bias, error, naive_emb

    # data checks: general
    cfg['data::normalise'] = 'z-score' #None
    cfg['data::error_norm'] = False
    cfg['data::past-include'] = False
    cfg['model::prediction_temporal:include_Y_past'] = False
    cfg['data:train:sparse'] = None
    cfg['data:test:sparse'] = None                  # None = full dataset used ie. daily measurements
    cfg['data::previously_calibrated'] = False                          # Call preprocessed Datasets


    #train
    cfg['data::train::station_id']= { # '1': {'startdate':[[2024,5,13]], 'stopdate': [[2024,12,31]]}, 
                                        '21': {'startdate':[[2002,1,1]], 'stopdate': [[2004,12,31]]}
                                    } # Collelongo, Le Bray, Soro, Hyytiälä,
    #test
    cfg['data::test::station_id']= { # '1': {'startdate':[[2024,5,13]], 'stopdate': [[2024,12,31]]},
                                    '5': {'startdate':[[2004,1,1]], 'stopdate': [[2005,12,31]]}
                                    }

    cfg['model::windows_input']=None
    cfg['model::windows_output']=None


    # Parameters
    cfg['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    cfg['data::train::y_parameter']= ['GPP','ET']
    cfg['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    cfg['data::test::y_parameter']= ['GPP','ET']


    #preles parameters (maybe in another cfg)
    cfg['Preles::x_para'] =  ['PAR','Tair','VPD','Precip','fapar','CO2' ,'DOY'] #'CO2' ,'DOY'
    cfg['Preles::y_para'] = ['GPP', 'ET'] #
    cfg['Preles::y_phy_para'] = ['GPP_pred','ET_pred'] #'SW_pred'


    ### PRELES ###
    #Default Inputs
    cfg['Preles::call:in'] = {'soildepth': 413,'ThetaFC': 0.45,'ThetaPWP': 0.118,'tauDrainage': 3,'beta': 0.7457,'tau': 10.93,
                            'S0': -3.063,'Smax': 17.72,'kappa': -0.1027,'gamma': 0.03673,'soilthres': 0.7779,'bCO2': 0.5,'xCO2': -0.364,'ETbeta': 0.2715,
                            'ETkappa': 0.8351,'ETchi': 0.07348,'ETsoilthres': 0.9996,'ETnu': 0.4428,'MeltCoef': 1.2,'I0': 0.33,'CWmax': 4.970496,'SnowThreshold': 0.0,
                            'T_0': 0.0,'SWinit': 160.0,'CWinit': 0.0,'SOGinit': 0.0,'Sinit': 0.0,'t0': -999.0,'tcrit': -999.0,'tsumcrit': -999.0,
                            'GPPmeas': -999.0,'ETmeas': -999.0,'SWmeas': -999.0,'transp': -999.0,'evap': -999.0,'fWE': -999.0
    }

    #PRELES variables to tune: 5:11, 14:18, 31 
    cfg['Preles::parameters:to:tune'] = {
        'soildepth': {'value': 413, 'min': 100, 'max': 1300},                # soildepth in mm
        'tauDrainage': {'value': 3, 'min': 3, 'max': 20},                    # Time constant for drainage/percolation below FC; controls how quickly excess soil water leaves the bucket
        
        ## GPP_MODEL_PARAMETERS
        'beta':     {'value': 0.748018, 'min': 0.2, 'max': 2.5},            # betaGPP
        'tau':      {'value': 13.23383, 'min': 1, 'max': 35},               # tau
        'S0':       {'value': -3.9657867, 'min': -30, 'max': 30.1},         # S0     #22/05/25, increased from -20 and 20.
        'Smax':     {'value': 18.76696, 'min': 2.3, 'max': 45},             # Smax
        'kappa':    {'value': -0.130473, 'min': -1, 'max': -0.001},         # kappa
        'gamma':    {'value': 0.034459, 'min': 1e-05, 'max': 0.503},        # gamma
        'soilthres': {'value': 0.7779, 'min': 0, 'max': 1},                 # soilthres
        
        ## EVAPOTRANSPIRATION_PARAMETERS
        'ETbeta': {'value': 0.2715, 'min': 1e-06, 'max': 10},               # ETbeta
        'ETkappa': {'value': 0.8351, 'min': 0.0001, 'max': 2},              # ETkappa
        'ETchi': {'value': 0.07348, 'min': 0, 'max': 2.5},                  # ETchi
        'ETsoilthres': {'value': 0.9996, 'min': 0, 'max': 0.9999},          # ETsoilthres
        'ETnu': {'value': 0.4428, 'min': 0.0001, 'max': 7.5},               # ETnu
        
        ## SITE SPECIFIC
        
    } 
    
    #Outputs
    cfg['Preles::out'] = {'GPP': 0.0, 'ET': 0.0, 'SW': 0.0, 'SOG': 0.0, 'fS': 0, 'fD': 0.0, 'fW': 0.0, 'fE': 0.0, 'Throughfall': 0.0, 
                        'Interception': 0, 'Snowmelt': 0, 'Drainage': 0, 'Canopywater': 0, 'S': 0}

    cfg['Preles::control_arg'] = {'LOGFLAG':0, 'etmodel':0, 'NofDays': 1}

    cfg['Preles::preintialised_output'] = {'SW':0, 'SOG':0,'fS':0, 'fD':0, 'fW':0,'fE': 0.,'Throughfall':0.,'Interception': 0,
                                        'Snowmelt': 0.,'Drainage':0, 'Canopywater': 0., 'S': 0.}

    cfg['Preles::possible_preles_keys'] ={'PAR':'PAR','TAir':'TAir', 'VPD' :'VPD', 'Precip' : 'Precip', 'CO2' : 'CO2', 'fAPAR' : 'fAPAR', 
                                        'GPPmeas' : 'GPPmeas', 'ETmeas' : 'ETmeas', 'SWmeas': 'SWmeas', 
                                        'GPP': 'GPP', 'ET': 'ET', 'SW': 'SW', 'SOG': 'SOG', 'fS': 'fS', 'fD': 'fD', 'fW': 'fW', 'fE': 'fE', 'Throughfall': 'Throughfall', 
                                        'Interception': 'Interception', 'Snowmelt': 'Snowmelt', 'Drainage': 'Drainage', 'Canopywater': 'Canopywater', 'S': 'S', 'soildepth': 'p1', 
                                        'ThetaFC': 'p2', 'ThetaPWP': 'p3', 'tauDrainage': 'p4', 'beta': 'p5', 'tau': 'p6', 'S0': 'p7', 'Smax': 'p8', 'kappa': 'p9', 'gamma': 'p10', 'soilthres': 'p11', 
                                        'bCO2': 'p12', 'xCO2': 'p13', 'ETbeta': 'p14', 'ETkappa': 'p15', 'ETchi': 'p16', 'ETsoilthres': 'p17', 'ETnu': 'p18', 'MeltCoef': 'p19', 'I0': 'p20', 'CWmax': 'p21', 
                                        'SnowThreshold': 'p22', 'T_0': 'p23', 'SWinit': 'p24', 'CWinit': 'p25', 'SOGinit': 'p26', 'Sinit': 'p27', 't0': 'p28', 'tcrit': 'p29', 'tsumcrit': 'p30', 
                                        'etmodel': 'control', 'LOGFLAG': 'logflag', 'NofDays': 'NofDays', 'day': 'DOY', 'transp': 'transp', 'evap': 'evap', 'fWE': 'fWE'}

    cfg['Preles::parameter_order'] = [
                            # drivers + measurements (time-varying)
                            'PAR', 'Tair', 'VPD', 'Precip', 'CO2', 'fapar', 'GPPmeas', 'ETmeas', 'SWmeas', 
                            'GPP', 'ET', 'SW', 'SOG', 'fS', 'fD', 'fW', 'fE', 'Throughfall', 
                            'Interception', 'Snowmelt', 'Drainage', 'Canopywater', 'S', 
                            
                            # site/process parameters (scalars per run)
                            'soildepth',        # Effective root-zone depth that stores liquid water #How to set: rooting depth × bulk depth; common 300–1200 mm depending on vegetation/soil.
                            'ThetaFC',          # volumetric water content at field capacity #How to set: from soil texture/BDC or lab/PTS tables.
                            'ThetaPWP',         # permanent wilting point (PWP) #How to set: from soil texture/PTS tables.
                            'tauDrainage',      # Time constant for drainage/percolation below FC; controls how quickly excess soil water leaves the bucket
                            'beta',             # Base light-use efficiency coefficient in the GPP equation
                            'tau',              # Model-specific time/shape constant used in the physiology block
                            'S0',               # Initial value of the seasonal acclimation state S at simulation start
                            'Smax',             # Upper bound / asymptote for S (peak season)
                            'kappa',            # Shape/rate parameter for the seasonal acclimation (state S) tune vs. seasonal GPP rise/decay.
                            'gamma',            # GPP modifier
                            'soilthres',        # Water-stress threshold for photosynthesis/transpiration modifier
                            'bCO2',             # CO₂ fertilization function parameters
                            'xCO2', 
                            'ETbeta', 
                            'ETkappa', 
                            'ETchi', 
                            'ETsoilthres', 
                            'ETnu', 
                            'MeltCoef',         # Degree-day snowmelt coefficient
                            'I0',               # Canopy interception efficiency/curvature parameter
                            'CWmax',            # Maximum canopy water storage capacity; can be estimated from LAI - 
                            'SnowThreshold',    # Air-temperature cutoff between snow and rain
                            'T_0',              # Base (freezing) temperature for degree-day melt formulation
                            
                            # initial storages (scalars; t0 initial conditions)
                            'SWinit', 'CWinit', 'SOGinit', 'Sinit',  #Snow on Ground , # State of temperature acclimation
                            # phenology/controls
                            't0', 'tcrit', 'tsumcrit', 'etmodel', 'LOGFLAG', 'NofDays', 'DOY', 'transp', 'evap', 'fWE'
                            ]
    
    #hyper_tuning
    cfg['checkpoint::window_size'] = 10

    # Get project root from environment
    root = os.getenv("PROJECT_ROOT", os.getcwd())

    def env_or_default(env_key, fallback_subdir):
        return os.getenv(env_key, os.path.join(root, fallback_subdir))

    # Assign paths with override option if available in environment file, otherwise defaults
    cfg['cfg::save'] = env_or_default("CFG_SAVE_PATH", "")
    cfg['checkpoint::path_dir'] = env_or_default("CHECKPOINT_PATH", "checkpoints")
    data_path = env_or_default("DATA_PATH", "Data")
    cfg['data::train::path_dir'] = data_path
    cfg['data::test::path_dir'] = data_path

    # Canonical calibration/path keys for process-model-specific storage (as config_spatial.py)
    cfg['process_model::name'] = cfg.get('process_model::name', 'preles')
    cfg['calibration::mode'] = cfg.get('calibration::mode', 'auto')  # defaults | auto | force_calibrate
    cfg['calibration::root'] = env_or_default("CALIBRATION_ROOT", "Data/calibrations")
    cfg['calibration::cv_type'] = cfg.get('model::prediction_type', 'spatial') #default or temporal_window

    return cfg

def save_cfg(config, name):
    '''saves dictionary'''
    os.makedirs(config['cfg::save'], exist_ok=True)
    path = os.path.join(config['cfg::save'], name + ".json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Config saved to {path}")
    return path

def json_open(path):
    '''opens json'''
    with open(path) as json_file:
        dict_ = json.load(json_file)
    return dict_

def conv_torch(data,dtype=64,grad=True):
    return torch.tensor(data,dtype=dtype,requires_grad=grad)

if __name__=='__main__':
    config = build_config()   # Config is now a dynamic build, using pathways from local env. file set by user
    print(f"[INFO] Using config with root: {config['cfg::save']}")

    ### Optional: save snapshot ###
    save_cfg(config, 'config_spatial') 
    path= config['cfg::save'] + '/config_spatial.json'

    config=json_open(path)
    print('saved')
    



    



