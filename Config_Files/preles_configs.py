### PRELES ###
#Default Inputs
cfg= {}

cfg['data::train::station_id']= {
                                'DK-Sor': {'startdate': [[2006, 1, 1]],'stopdate': [[2006, 12, 31]],
                                    #'source_tag': 'flx',   # or 'icos' if none then flx
                                    },
                                'FI-Hyy': {'startdate': [[2010, 1, 1]],'stopdate': [[2010, 12, 31]],
                                           },
                                'IT-Col': {'startdate': [[2012, 1, 1]],'stopdate': [[2012, 12, 31]],
                                }
    }

#test
cfg['data::test::station_id']= { # '1': {'startdate':[[2024,5,13]], 'stopdate': [[2024,12,31]]},
                                'FR-LBr': {'startdate':[[2006,1,1]], 'stopdate': [[2008,12,31]]}
                                }

cfg['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
cfg['data::train::y_parameter']= ['GPP','ET']
cfg['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar']
cfg['data::test::y_parameter']= ['GPP','ET']

cfg['Preles::x_para'] =  ['PAR','Tair','VPD','Precip','fapar','CO2' ,'DOY'] #'CO2' ,'DOY'
cfg['Preles::y_para'] = ['GPP', 'ET'] #
cfg['Preles::y_phy_para'] = ['GPP_pred','ET_pred', 'SWC_pred'] #'SW_pred'
cfg["model::io::y_pred_additional_cols"] = [] #[]

cfg['Preles::call:in'] = {'soildepth': 413,'ThetaFC': 0.45,'ThetaPWP': 0.118,'tauDrainage': 3,'beta': 0.7457,'tau': 10.93,
                        'S0': -3.063,'Smax': 17.72,'kappa': -0.1027,'gamma': 0.03673,'soilthres': 0.7779,'bCO2': 0.5,'xCO2': -0.364,'ETbeta': 0.2715,
                        'ETkappa': 0.8351,'ETchi': 0.07348,'ETsoilthres': 0.9996,'ETnu': 0.4428,'MeltCoef': 1.2,'I0': 0.33,'CWmax': 4.970496,'SnowThreshold': 0.0,
                        'T_0': 0.0,'SWinit': 160.0,'CWinit': 0.0,'SOGinit': 0.0,'Sinit': 0.0,'t0': -999.0,'tcrit': -999.0,'tsumcrit': -999.0,
                        'GPPmeas': -999.0,'ETmeas': -999.0,'SWmeas': -999.0,'transp': -999.0,'evap': -999.0,'fWE': -999.0
}

#PRELES variables to tune: 5:11, 14:18, 31 
cfg['Preles::parameters:to:tune'] = {
    'soildepth': {'value': 413, 'min': 100, 'max': 1000},                 # soildepth in mm
    'tauDrainage': {'value': 3, 'min': 0.5, 'max': 8},                    # Time constant for drainage/percolation below FC; controls how quickly excess soil water leaves the bucket
    'ThetaPWP':    {'value': 0.12, 'min': 0.04, 'max': 0.25},
    # Enforce FC > PWP via delta parameterization in the sampler
    # or set a broad but valid range if you can’t encode constraints:
    'ThetaFC':     {'value': 0.45, 'min': 0.25, 'max': 0.50},
    
    ## GPP_MODEL_PARAMETERS
    'beta':     {'value': 0.748018, 'min': 0.3, 'max': 1.5},            # betaGPP
    'tau':      {'value': 13.23383, 'min': 1, 'max': 35},               # tau
    'S0':       {'value': -3.9657867, 'min': -15, 'max': 15},           # S0     #22/05/25, increased from -20 and 20.
    'Smax':     {'value': 18.76696, 'min': 2.3, 'max': 250},            # Smax   #27/08/25, increased from 45 to 50. # Upper bound / asymptote for S (peak season)
    'kappa':    {'value': -0.130473, 'min': -1, 'max': -0.001},         # kappa     VPD shape
    'gamma':    {'value': 0.034459, 'min': 1e-05, 'max': 0.503},        # gamma
    'soilthres': {'value': 0.7779, 'min': 0, 'max': 1},                 # soilthres
    
    ## EVAPOTRANSPIRATION_PARAMETERS
    'ETbeta': {'value': 0.2715, 'min': 1e-06, 'max': 10},               # ETbeta
    'ETkappa': {'value': 0.8351, 'min': 0.003, 'max': 1.5},             # ETkappa
    'ETchi': {'value': 0.07348, 'min': 0, 'max': 2.5},                  # ETchi
    'ETsoilthres': {'value': 0.5, 'min': 0, 'max': 0.9999},             # ETsoilthres   27/05/25, changed from 0.9996 soil stress throttling
    'ETnu': {'value': 0.4428, 'min': 0.0001, 'max': 7.5},               # ETnu
    
    ## SITE SPECIFIC
    
} 

#Outputs
cfg['Preles::out'] = {'GPP': 0.0, 'ET': 0.0, 'SWC': 0.0, 'SOG': 0.0, 'fS': 0, 'fD': 0.0, 'fW': 0.0, 'fE': 0.0, 'Throughfall': 0.0, 
                    'Interception': 0, 'Snowmelt': 0, 'Drainage': 0, 'Canopywater': 0, 'S': 0}

cfg['Preles::control_arg'] = {'LOGFLAG':0, 'etmodel':0, 'NofDays': 1}

# PRELES execution backend:
#   'preles_c'       -> batched C extension; soil-water/snow/acclimation state is
#                       carried forward across days internally (default).
#   'preles_rewrite' -> torch rewrite called row-by-row; each day re-seeds its
#                       initial states from the constants here (NO carry-forward).
cfg['Preles::backend'] = 'preles_c'

# Inner-HPO budget per outer fold for the PRELES (paper-1) NCV runs, split across the
# fold's workers: each worker runs ceil(trials_per_fold / workers_per_fold) trials.
# Read by Optuna_{spatial,temporal}_NCV.py; an explicit --n_trials (per worker) overrides it.
cfg['hpo::trials_per_fold'] = 48

cfg['Preles::preintialised_output'] = {'SWC':0, 'SOG':0,'fS':0, 'fD':0, 'fW':0,'fE': 0.,'Throughfall':0.,'Interception': 0,
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
                        'soildepth',        # Effective root-zone depth that stores liquid water    #How to set: rooting depth × bulk depth; common 300–1200 mm depending on vegetation/soil.
                        'ThetaFC',          # volumetric water content at field capacity            #How to set: from soil texture/BDC or lab/PTS tables.
                        'ThetaPWP',         # permanent wilting point (PWP)                         #How to set: from soil texture/PTS tables.
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
