# config_spatial.py

import json
import torch
import os
from utilis_shorts import save_json, json_open
from dotenv import load_dotenv

load_dotenv()

def build_config():
    cfg= {}

    # Model architecture
    cfg['model::modeltype'] =  'MLP' #MLP' #'Adapted_error' # 'MLP' #'LSTM 'Adapted_error'

    # For MLP
    cfg['model::layer_number']= 3 #MLP Layer number
    cfg['model::input_size']= 6
    cfg['model::output_size']= 3
    cfg['model::hidden_size'] = [30,40]
    cfg['model::activation'] =  ['Relu','Relu', None] #* cfg['model::layer_number']  #'Tanh', None
    cfg['model::dropout_list'] = 0.1 #* cfg['model::layer_number']
    cfg['model::bias_list'] = [True] * cfg['model::layer_number'] #[True, True, True]
    cfg['model::save_path'] = None
    cfg['model::seed'] = 42 

    # Model training
    cfg['training::cuda'] = True
    cfg['training::learning_rate']= 0.00001
    cfg['training::momentum']= 0 #SGD
    cfg['training::weight_decay'] = 0
    cfg['training::dampening']= 0 #SGD
    cfg['training::beta_1']= 0.9 #Adam
    cfg['training::beta_2']= 0.999 #Adam
    cfg['training::optimizer'] = 'Adam' #AdamW, SGD ...
    cfg['training::loss_type'] = 'MSE' #'MAE' #MSE' #MA
    cfg['training::physical_loss_regul_term']= 0.2
    cfg['training::epochs'] = 100              #Inner loops
    cfg['training::seed'] = 42
    cfg['training::acc_metric'] = 'MAE' #'R2','MSE'#
    cfg['training::batchsize'] = 16
    cfg['data::test::batchsize'] = 16

    # --- Inner loop: divergence abort -------------------------------------------
    # A non-finite TRAIN loss means the weights are non-finite and cannot recover, so
    # the trial is pruned immediately rather than running out its epoch budget and
    # being recorded as FAIL. patience = consecutive non-finite epochs before aborting.
    cfg['training::divergence_abort::enabled'] = True
    cfg['training::divergence_abort::patience'] = 1

    # --- Outer loop: convergence stop -------------------------------------------
    # Deliberately on TRAINING loss: the outer run is a retrain of an already-chosen
    # architecture, so it needs a convergence criterion, not model selection, and must
    # never look at the test split. `outerepochs` remains the hard backstop.
    cfg['training::outer_early_stop::enabled'] = True
    cfg['training::outer_early_stop::patience'] = 20
    cfg['training::outer_early_stop::min_delta'] = 1e-4   # relative

    # --- Outer loop: LR schedule --------------------------------------------------
    # ReduceLROnPlateau on train loss. Keep `patience` well below the early-stop
    # patience above, or the run stops before a reduction can take effect.
    cfg['training::outer_lr_schedule::enabled'] = True
    cfg['training::outer_lr_schedule::factor'] = 0.5
    cfg['training::outer_lr_schedule::patience'] = 8
    cfg['training::outer_lr_schedule::min_lr'] = 1e-7

    # Experimental Version 
    cfg['model::prediction_type']= 'spatial'            #temporal' #'spatial'
    cfg['training::loss_physic_type'] = 'naive'      #'naive' # parallel, regularized, #bias, error, naive_emb

    # data checks: general
    cfg['data::normalise'] = 'z-score' #None
    cfg['data::error_norm'] = False
    cfg['data::past-include'] = False
    cfg['model::prediction_temporal:include_Y_past'] = False
    cfg['data:train:sparse'] = None
    cfg['data:test:sparse'] = None                  # None = full dataset used ie. daily measurements
    cfg['data::previously_calibrated'] = False                          # Call preprocessed Datasets


    # #train
    # cfg['data::train::station_id']= { # '1': {'startdate':[[2024,5,13]], 'stopdate': [[2024,12,31]]}, 
    #                                 'AU-Wom': {'startdate':[[2011,1,1]], 'stopdate': [[2012,12,31]]} #out_l_3
    #                                 } # Collelongo, Le Bray, Soro, Hyytiälä,
    # #test
    # cfg['data::test::station_id']= { # '1': {'startdate':[[2024,5,13]], 'stopdate': [[2024,12,31]]},
    #                                 'FR-LBr': {'startdate':[[2007,1,1]], 'stopdate': [[2008,12,31]]}
    #                                 }

    cfg['model::windows_input']=None
    cfg['model::windows_output']=None


    # # Parameters
    # cfg['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    # cfg['data::train::y_parameter']= ['GPP','ET']
    # cfg['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    # cfg['data::test::y_parameter']= ['GPP','ET']

    # #Bias 
    # if cfg['training::loss_physic_type'] == 'bias':
    #     cfg['model::input_size']= 2
    #     cfg['data::train::x_parameter']= ['GPP_pred', 'ET_pred'] #, 'SW_pred'
    #     cfg['data::test::x_parameter']= ['GPP_pred', 'ET_pred']
    
    # #Residual    
    # if cfg['training::loss_physic_type'] == 'residual':
    #     cfg['model::input_size']= 7 #preles specific, needs update for basfor
    #     cfg['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar','GPP_pred', 'ET_pred'] #, 'SW_pred'
    #     cfg['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar','GPP_pred', 'ET_pred']


    # #Process Model parameters (maybe in another cfg)
    # cfg['Process::x_para'] =  ['PAR','Tair','VPD','Precip','fapar','CO2' ,'DOY'] #'CO2' ,'DOY'
    # cfg['Process::y_para'] = ['GPP', 'ET'] #
    # cfg['Process::y_phy_para'] = ['GPP_pred','ET_pred'] #'SW_pred'

    # if cfg['model::modeltype'] ==  'embedding':
    #     cfg['training::batchsize'] = 1
    #     cfg['data::test::batchsize'] = 1

    #     #additional parameters for preles
    #     cfg['Preles::add_para']={
    #             'etmodel': 0, 
    #             'LOGFLAG': 0,  # Changed to tensor
    #             'NofDays': 1}
    
    #     cfg['Preles::preles_out_arg']= [0,1]

    #     #initialised parameter
    #     cfg['Preles::intilaised para'] = {
    #             'soildepth': {'value':413., 'grad':True}, #torch.rand((1), requires_grad=True), # torch.tensor(413., requires_grad=True),
    #             'ThetaFC':  {'value':0.45, 'grad':False}, #torch.tensor(0.45, requires_grad=False),
    #             'ThetaPWP':  {'value':0.118, 'grad':False}, #torch.tensor(0.118, requires_grad=False),
    #             'tauDrainage': {'value':3., 'grad':False}, #torch.tensor(3., requires_grad=False),
    #             'beta':{'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(0.7457, requires_grad=True),
    #             'tau': {'value':10.93, 'grad':True}, #torch.tensor(10.93, requires_grad=True), #torch.rand((1), requires_grad=True), #torch.tensor(10.93, requires_grad=True),
    #             'S0': {'value':-3.063, 'grad':True}, #torch.tensor(-3.063, requires_grad=True),#torch.rand((1), requires_grad=True), #,torch.tensor(-3.063, requires_grad=True),
    #             'Smax':{'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #,torch.tensor(17.72, requires_grad=True),
    #             'kappa': {'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(-0.1027, requires_grad=True),
    #             'gamma': {'value':'rand_norm', 'grad':True},#torch.rand((1), requires_grad=True), # torch.tensor(0.03673, requires_grad=True),
    #             'soilthres': {'value':0.7779, 'grad':False}, #torch.tensor(0.7779, requires_grad=False),
    #             'bCO2': {'value':0.5, 'grad':False}, #torch.tensor(0.5, requires_grad=False),
    #             'xCO2':   {'value':-0.364, 'grad':False},  #torch.tensor(-0.364, requires_grad=False),
    #             'ETbeta': {'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(0.2715, requires_grad=True),
    #             'ETkappa': {'value':'rand_norm', 'grad':True},#torch.rand((1), requires_grad=True), #torch.tensor(0.8351, requires_grad=True),
    #             'ETchi': {'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(0.07348, requires_grad=True),
    #             'ETsoilthres': {'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(0.9996, requires_grad=True),
    #             'ETnu': {'value':'rand_norm', 'grad':True}, #torch.rand((1), requires_grad=True), #torch.tensor(0.4428, requires_grad=True),
    #             'MeltCoef': {'value':1.2, 'grad':False}, #torch.tensor(1.2, requires_grad=False),
    #             'I0': {'value':0.33, 'grad':False},  #torch.tensor(0.33, requires_grad=False),
    #             'CWmax': {'value':4.970496, 'grad':False} ,#torch.tensor(4.970496, requires_grad=False),
    #             'SnowThreshold': {'value':4.970496, 'grad':False},#torch.tensor(0., requires_grad=False),
    #             'T_0': {'value':0., 'grad':False}, #torch.tensor(0., requires_grad=False),
    #             'SWinit': {'value':160., 'grad':False}, #torch.tensor(160., requires_grad=False),
    #             'CWinit': {'value':0., 'grad':False}, #torch.tensor(0., requires_grad=False),
    #             'SOGinit': {'value':0., 'grad':False}, #torch.tensor(0., requires_grad=False),
    #             'Sinit': {'value':0., 'grad':False}, #torch.tensor(0.0, requires_grad=False),  # Optional: Not trainable
    #             't0': {'value':-999., 'grad':False}, #torch.tensor(-999., requires_grad=False),
    #             'tcrit': {'value':-999., 'grad':False}, #torch.tensor(-999., requires_grad=False),
    #             'tsumcrit': {'value':-999., 'grad':False}, } #torch.tensor(-999., requires_grad=False),}

    #     ##MLP
    #     cfg['Preles::output_MLP_layer']=None 
    #     if cfg['Preles::output_MLP_layer']!=None:
    #         cfg['layer_number_out'] = 3
    #         cfg['input_size_out']= 2
    #         cfg['output_size_out']= 2
    #         cfg['hidden_size_out']= [3,2]
    #         cfg['activation_out']= ['Relu', 'Relu', None]
    #         cfg['dropout_list_out']= 0.1
    #         cfg['bias_list_out']= [True]*cfg['layer_number_out']

    #     #MLP optimizers #can diverge from Preles which could enforce Preles to learn parameters
    #     cfg['output_MLP:: Learning_rate'] = 0.00001

    # if cfg['model::modeltype'] ==  'Adapted_error': #add additional model to learn the error
    #     cfg['model::prediction_type']= 'temporal' 
    #     cfg['model:real::modeltype'] = 'MLP'
    #     cfg['training::loss_physic_type'] = 'error'

    #     cfg['model::layer_number']= 4 #MLP Layer number
    #     cfg['model::input_size']= 5
    #     cfg['model::output_size']= 1 
    #     cfg['model::hidden_size'] = [40,20,10]
    #     cfg['model::activation'] =  ['Relu','Relu', 'Relu', None] #* cfg['model::layer_number']  #'Tanh', None
    #     cfg['model::dropout_list'] = [0.2] #* cfg['model::layer_number']
    #     cfg['model::save_path'] = None
    #     cfg['model::seed'] = 42 

    #     cfg['model:error::modeltype'] = 'MLP' #for error model
    #     cfg['model:error::layer_number']= 3
    #     cfg['model:error::input_size']= 2
    #     cfg['model:error::output_size']= 2 
    #     cfg['model:error::hidden_size'] = [20,30]
    #     cfg['model:error::activation'] =  ['Relu','Relu', None] #* cfg['model::layer_number']  #'Tanh', None
    #     cfg['model:error::dropout_list'] = [0.2] #* cfg['model::layer_number']
    #     cfg['model:error::bias_list'] = [True] * cfg['model::layer_number'] #[True, True, True] 
    #     cfg['model:error::save_path'] = None
    #     cfg['model:error::seed'] = 42

    #     cfg['training:error::learning_rate']= 0.001
    #     cfg['training:error::momentum']= 0 #SGD
    #     cfg['training:error::weight_decay'] = 0
    #     cfg['training:error::dampening']= 0 #SGD
    #     cfg['training:error::beta_1']= 0.9 #Adam
    #     cfg['training:error::beta_2']= 0.999 #Adam
    #     cfg['training:error::optimizer'] = 'Adam' #Adam ...
    #     cfg['training:error::loss_type'] = 'MAE' #'MSE''MAE'...
    #     cfg['training::error:batchsize'] = 1
    #     cfg['training:error::acc_metric'] = 'MSE'#'MAE' #'R2','MAE'#

    #     # Parameters
    #     cfg['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    #     cfg['data::train::y_parameter']= ['GPP']#,'ET']
    #     cfg['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
    #     cfg['data::test::y_parameter']= ['GPP'] #,'ET']
    #     cfg['data::test::batchsize'] = 1

    #     cfg['Preles::x_para'] =  ['PAR','Tair','VPD','Precip','fapar'] #'CO2' ,'DOY'
    #     cfg['Preles::y_para'] = ['GPP'] #'ET#
    #     cfg['Preles::y_phy_para'] = ['GPP_pred'] #, 'ET_pred'] #'SW_p

    # if cfg['model::prediction_type']=='temporal':
    #     #Temporal Runs
    #     cfg['model::windows_input']=10
    #     cfg['model::windows_output']=5

    #     #rewrite input and output size depending on window size
    #     cfg['model::input_size']=  (cfg['model::input_size'])* cfg['model::windows_input'] #+cfg['model::output_size'] #include if you want to use past predictions
    #     cfg['model::output_size']=  cfg['model::output_size']*cfg['model::windows_output']
    #     # mat2 

    #     # #train
    #     # cfg['data::train::station_id']= [12] # Hyytiälä, Collelongo, Le Bray, Soro 
    #     # cfg['data::train::startdate']= [2008,1,1] #[[2004,1,1], [2008,1,1]] #date: year, month, day or list of different start dates
    #     # cfg['data::train::stopdate']= [2011,12,31] #[[2005,12,31], [2012,12,31]]  #date: year, month, day
        
    #     # # cfg['data::train::startdate']= [2004,1,1] #date: year, month, day
    #     # # cfg['data::train::stopdate']= [2005,12,31]  #date: year, month, day

    #     # #test
    #     # cfg['data::test::station_id']= [12] # Hyytiälä, Collelongo, Le Bray, Soro
    #     # cfg['data::test::startdate']= [2012,1,1]   #date: year, month, day
    #     # cfg['data::test::stopdate']= [2012,12,31]  #date: year, month, day
    
    #     if cfg['training::loss_physic_type'] == 'error' and cfg['model::modeltype'] ==  'Adapted_error':
    #         cfg['model:error::input_size'] = (len(cfg['Preles::x_para']) + len(cfg['Preles::y_phy_para']))*cfg['model::windows_input']
    #         cfg['model:error::output_size'] = len(cfg['Preles::y_phy_para'])* cfg['model::windows_output']
    #         cfg['model::input_size']=  (len(cfg['Preles::x_para']))* cfg['model::windows_input']

    #     elif cfg['training::loss_physic_type'] == 'error' and cfg['model::modeltype'] !=  'Adapted_error':
    #         cfg['model::input_size'] = (len(cfg['Preles::x_para']) + len(cfg['Preles::y_phy_para']))*cfg['model::windows_input']
    #         cfg['model::output_size'] = len(cfg['Preles::y_phy_para'])* cfg['model::windows_output']

    #     if cfg['model::modeltype'] ==  'LSTM':
    #         cfg['model::lstm_depth'] = 1
    #         cfg['model::hidden_lstm'] = 3
    #         cfg['model::input_size']= len(cfg['data::train::x_parameter']) + len(cfg['data::train::y_parameter'])

    
    # # ### PRELES ###
    # #Default Inputs
    # cfg['Preles::call:in'] = {'soildepth': 413,'ThetaFC': 0.45,'ThetaPWP': 0.118,'tauDrainage': 3,'beta': 0.7457,'tau': 10.93,
    #                         'S0': -3.063,'Smax': 17.72,'kappa': -0.1027,'gamma': 0.03673,'soilthres': 0.7779,'bCO2': 0.5,'xCO2': -0.364,'ETbeta': 0.2715,
    #                         'ETkappa': 0.8351,'ETchi': 0.07348,'ETsoilthres': 0.9996,'ETnu': 0.4428,'MeltCoef': 1.2,'I0': 0.33,'CWmax': 4.970496,'SnowThreshold': 0.0,
    #                         'T_0': 0.0,'SWinit': 160.0,'CWinit': 0.0,'SOGinit': 0.0,'Sinit': 0.0,'t0': -999.0,'tcrit': -999.0,'tsumcrit': -999.0,
    #                         'GPPmeas': -999.0,'ETmeas': -999.0,'SWmeas': -999.0,'transp': -999.0,'evap': -999.0,'fWE': -999.0
    # }

    # #PRELES variables to tune: 5:11, 14:18, 31 
    # cfg['Preles::parameters:to:tune'] = {
    #     'soildepth': {'value': 413, 'min': 100, 'max': 1000},                # soildepth in mm
    #     'tauDrainage': {'value': 3, 'min': 0.5, 'max': 8},                    # Time constant for drainage/percolation below FC; controls how quickly excess soil water leaves the bucket
    #     'ThetaPWP':    {'value': 0.12, 'min': 0.04, 'max': 0.25},
    #     # Enforce FC > PWP via delta parameterization in the sampler
    #     # or set a broad but valid range if you can’t encode constraints:
    #     'ThetaFC':     {'value': 0.45, 'min': 0.25, 'max': 0.50},
        
    #     ## GPP_MODEL_PARAMETERS
    #     'beta':     {'value': 0.748018, 'min': 0.3, 'max': 1.5},            # betaGPP
    #     'tau':      {'value': 13.23383, 'min': 1, 'max': 35},               # tau
    #     'S0':       {'value': -3.9657867, 'min': -15, 'max': 15},           # S0     #22/05/25, increased from -20 and 20.
    #     'Smax':     {'value': 18.76696, 'min': 2.3, 'max': 250},            # Smax   #27/08/25, increased from 45 to 50. # Upper bound / asymptote for S (peak season)
    #     'kappa':    {'value': -0.130473, 'min': -1, 'max': -0.001},         # kappa     VPD shape
    #     'gamma':    {'value': 0.034459, 'min': 1e-05, 'max': 0.503},        # gamma
    #     'soilthres': {'value': 0.7779, 'min': 0, 'max': 1},                 # soilthres
        
    #     ## EVAPOTRANSPIRATION_PARAMETERS
    #     'ETbeta': {'value': 0.2715, 'min': 1e-06, 'max': 10},               # ETbeta
    #     'ETkappa': {'value': 0.8351, 'min': 0.003, 'max': 1.5},              # ETkappa
    #     'ETchi': {'value': 0.07348, 'min': 0, 'max': 2.5},                  # ETchi
    #     'ETsoilthres': {'value': 0.5, 'min': 0, 'max': 0.9999},             # ETsoilthres   27/05/25, changed from 0.9996 soil stress throttling
    #     'ETnu': {'value': 0.4428, 'min': 0.0001, 'max': 7.5},               # ETnu
        
    #     ## SITE SPECIFIC
        
    # } 
    
    # #Outputs
    # cfg['Preles::out'] = {'GPP': 0.0, 'ET': 0.0, 'SW': 0.0, 'SOG': 0.0, 'fS': 0, 'fD': 0.0, 'fW': 0.0, 'fE': 0.0, 'Throughfall': 0.0, 
    #                     'Interception': 0, 'Snowmelt': 0, 'Drainage': 0, 'Canopywater': 0, 'S': 0}

    # cfg['Preles::control_arg'] = {'LOGFLAG':0, 'etmodel':0, 'NofDays': 1}

    # cfg['Preles::preintialised_output'] = {'SW':0, 'SOG':0,'fS':0, 'fD':0, 'fW':0,'fE': 0.,'Throughfall':0.,'Interception': 0,
    #                                     'Snowmelt': 0.,'Drainage':0, 'Canopywater': 0., 'S': 0.}

    # cfg['Preles::possible_preles_keys'] ={'PAR':'PAR','TAir':'TAir', 'VPD' :'VPD', 'Precip' : 'Precip', 'CO2' : 'CO2', 'fAPAR' : 'fAPAR', 
    #                                     'GPPmeas' : 'GPPmeas', 'ETmeas' : 'ETmeas', 'SWmeas': 'SWmeas', 
    #                                     'GPP': 'GPP', 'ET': 'ET', 'SW': 'SW', 'SOG': 'SOG', 'fS': 'fS', 'fD': 'fD', 'fW': 'fW', 'fE': 'fE', 'Throughfall': 'Throughfall', 
    #                                     'Interception': 'Interception', 'Snowmelt': 'Snowmelt', 'Drainage': 'Drainage', 'Canopywater': 'Canopywater', 'S': 'S', 'soildepth': 'p1', 
    #                                     'ThetaFC': 'p2', 'ThetaPWP': 'p3', 'tauDrainage': 'p4', 'beta': 'p5', 'tau': 'p6', 'S0': 'p7', 'Smax': 'p8', 'kappa': 'p9', 'gamma': 'p10', 'soilthres': 'p11', 
    #                                     'bCO2': 'p12', 'xCO2': 'p13', 'ETbeta': 'p14', 'ETkappa': 'p15', 'ETchi': 'p16', 'ETsoilthres': 'p17', 'ETnu': 'p18', 'MeltCoef': 'p19', 'I0': 'p20', 'CWmax': 'p21', 
    #                                     'SnowThreshold': 'p22', 'T_0': 'p23', 'SWinit': 'p24', 'CWinit': 'p25', 'SOGinit': 'p26', 'Sinit': 'p27', 't0': 'p28', 'tcrit': 'p29', 'tsumcrit': 'p30', 
    #                                     'etmodel': 'control', 'LOGFLAG': 'logflag', 'NofDays': 'NofDays', 'day': 'DOY', 'transp': 'transp', 'evap': 'evap', 'fWE': 'fWE'}

    # cfg['Preles::parameter_order'] = [
    #                         # drivers + measurements (time-varying)
    #                         'PAR', 'Tair', 'VPD', 'Precip', 'CO2', 'fapar', 'GPPmeas', 'ETmeas', 'SWmeas', 
    #                         'GPP', 'ET', 'SW', 'SOG', 'fS', 'fD', 'fW', 'fE', 'Throughfall', 
    #                         'Interception', 'Snowmelt', 'Drainage', 'Canopywater', 'S', 
                            
    #                         # site/process parameters (scalars per run)
    #                         'soildepth',        # Effective root-zone depth that stores liquid water #How to set: rooting depth × bulk depth; common 300–1200 mm depending on vegetation/soil.
    #                         'ThetaFC',          # volumetric water content at field capacity #How to set: from soil texture/BDC or lab/PTS tables.
    #                         'ThetaPWP',         # permanent wilting point (PWP) #How to set: from soil texture/PTS tables.
    #                         'tauDrainage',      # Time constant for drainage/percolation below FC; controls how quickly excess soil water leaves the bucket
    #                         'beta',             # Base light-use efficiency coefficient in the GPP equation
    #                         'tau',              # Model-specific time/shape constant used in the physiology block
    #                         'S0',               # Initial value of the seasonal acclimation state S at simulation start
    #                         'Smax',             # Upper bound / asymptote for S (peak season)
    #                         'kappa',            # Shape/rate parameter for the seasonal acclimation (state S) tune vs. seasonal GPP rise/decay.
    #                         'gamma',            # GPP modifier
    #                         'soilthres',        # Water-stress threshold for photosynthesis/transpiration modifier
    #                         'bCO2',             # CO₂ fertilization function parameters
    #                         'xCO2', 
    #                         'ETbeta', 
    #                         'ETkappa', 
    #                         'ETchi', 
    #                         'ETsoilthres', 
    #                         'ETnu', 
    #                         'MeltCoef',         # Degree-day snowmelt coefficient
    #                         'I0',               # Canopy interception efficiency/curvature parameter
    #                         'CWmax',            # Maximum canopy water storage capacity; can be estimated from LAI - 
    #                         'SnowThreshold',    # Air-temperature cutoff between snow and rain
    #                         'T_0',              # Base (freezing) temperature for degree-day melt formulation
                            
    #                         # initial storages (scalars; t0 initial conditions)
    #                         'SWinit', 'CWinit', 'SOGinit', 'Sinit',  #Snow on Ground , # State of temperature acclimation
    #                         # phenology/controls
    #                         't0', 'tcrit', 'tsumcrit', 'etmodel', 'LOGFLAG', 'NofDays', 'DOY', 'transp', 'evap', 'fWE'
    #                         ]
    
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

    # Canonical calibration/path keys for process-model-specific storage
    cfg['process_model::name'] = cfg.get('process_model::name', 'preles')
    cfg['calibration::mode'] = cfg.get('calibration::mode', 'auto')  # defaults | auto | force_calibrate
    cfg['calibration::root'] = env_or_default("CALIBRATION_ROOT", "Data/calibrations")
    cfg['calibration::cv_type'] = cfg.get('model::prediction_type', 'spatial') #default or temporal_window
    
    # if cfg['calibration::cv_type'] == 'spatial':
    #     cfg['calibration::training_years_len'] = os.getenv("spatial_training_years_len")
    # else:
    #     cfg['calibration::training_years_len'] = os.getenv("temporal_training_years_len1")

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
    



    


