import os
import torch
import preles
import pandas as pd

# Define the Preles model with manual gradient tracking
class PrelesModel:
    def __init__(self):
        # Define parameters as torch.Tensor with requires_grad=True
        self.params = {
            'soildepth': torch.tensor(413., requires_grad=True, dtype=torch.float64),
            'ThetaFC': torch.tensor(0.45, requires_grad=True, dtype=torch.float64),
            'ThetaPWP': torch.tensor(0.118, requires_grad=True, dtype=torch.float64),
            'tauDrainage': torch.tensor(3., requires_grad=True, dtype=torch.float64),
            'beta': torch.tensor(0.7457, requires_grad=True, dtype=torch.float64),
            'tau': torch.tensor(10.93, requires_grad=True, dtype=torch.float64),
            'S0': torch.tensor(-3.063, requires_grad=True, dtype=torch.float64),
            'Smax': torch.tensor(17.72, requires_grad=True, dtype=torch.float64),
            'kappa': torch.tensor(-0.1027, requires_grad=True, dtype=torch.float64),
            'gamma': torch.tensor(0.03673, requires_grad=True, dtype=torch.float64),
            'soilthres': torch.tensor(0.7779, requires_grad=True, dtype=torch.float64),
            'bCO2': torch.tensor(0.5, requires_grad=True, dtype=torch.float64),
            'xCO2': torch.tensor(-0.364, requires_grad=True, dtype=torch.float64),
            'ETbeta': torch.tensor(0.2715, requires_grad=True, dtype=torch.float64),
            'ETkappa': torch.tensor(0.8351, requires_grad=True, dtype=torch.float64),
            'ETchi': torch.tensor(0.07348, requires_grad=True, dtype=torch.float64),
            'ETsoilthres': torch.tensor(0.9996, requires_grad=True, dtype=torch.float64),
            'ETnu': torch.tensor(0.4428, requires_grad=True, dtype=torch.float64),
            'MeltCoef': torch.tensor(1.2, requires_grad=True, dtype=torch.float64),
            'I0': torch.tensor(0.33, requires_grad=True, dtype=torch.float64),
            'CWmax': torch.tensor(4.970496, requires_grad=True, dtype=torch.float64),
            'SnowThreshold': torch.tensor(0., requires_grad=True, dtype=torch.float64),
            'T_0': torch.tensor(0., requires_grad=True, dtype=torch.float64),
            'SWinit': torch.tensor(160., requires_grad=True, dtype=torch.float64),
            'CWinit': torch.tensor(0., requires_grad=True, dtype=torch.float64),
            'SOGinit': torch.tensor(0., requires_grad=True, dtype=torch.float64),
            'Sinit': torch.tensor(0.0, requires_grad=True, dtype=torch.float64),  # Optional: Not trainable
            't0': torch.tensor(-999., requires_grad=True, dtype=torch.float64),
            'tcrit': torch.tensor(-999., requires_grad=True, dtype=torch.float64),
            'tsumcrit': torch.tensor(-999., requires_grad=True, dtype=torch.float64),
        }

    def forward(self, **inputs):
        # Explicitly map parameters to preles arguments
        return preles.preles(
            PAR=inputs['PAR'], TAir=inputs['TAir'], VPD=inputs['VPD'], Precip=inputs['Precip'], CO2=inputs['CO2'],
            fAPAR=inputs['fAPAR'], 
            GPPmeas=inputs.get('GPPmeas', torch.tensor([-999.], dtype=torch.float64)),
            ETmeas=inputs.get('ETmeas', torch.tensor([-999.], dtype=torch.float64)),
            SWmeas=inputs.get('SWmeas', torch.tensor([-999.], dtype=torch.float64)),
            GPP=inputs['GPP'], ET=inputs['ET'], 
            SW=torch.tensor([0.], dtype=torch.float64),
            SOG=torch.tensor([0.], dtype=torch.float64), fS=torch.tensor([0.], dtype=torch.float64),
            fD=torch.tensor([0.], dtype=torch.float64), fW=torch.tensor([0.], dtype=torch.float64),
            fE=torch.tensor([0.], dtype=torch.float64), Throughfall=torch.tensor([0.], dtype=torch.float64),
            Interception=torch.tensor([0.], dtype=torch.float64), Snowmelt=torch.tensor([0.], dtype=torch.float64),
            Drainage=torch.tensor([0.], dtype=torch.float64), Canopywater=torch.tensor([0.], dtype=torch.float64),
            S=torch.tensor([0.], dtype=torch.float64),
            soildepth=self.params['soildepth'], ThetaFC=self.params['ThetaFC'], ThetaPWP=self.params['ThetaPWP'],
            tauDrainage=self.params['tauDrainage'], beta=self.params['beta'], tau=self.params['tau'],
            S0=self.params['S0'], Smax=self.params['Smax'], kappa=self.params['kappa'], gamma=self.params['gamma'],
            soilthres=self.params['soilthres'], bCO2=self.params['bCO2'], xCO2=self.params['xCO2'],
            ETbeta=self.params['ETbeta'], ETkappa=self.params['ETkappa'], ETchi=self.params['ETchi'],
            ETsoilthres=self.params['ETsoilthres'], ETnu=self.params['ETnu'], MeltCoef=self.params['MeltCoef'],
            I0=self.params['I0'], CWmax=self.params['CWmax'], SnowThreshold=self.params['SnowThreshold'],
            T_0=self.params['T_0'], SWinit=self.params['SWinit'], CWinit=self.params['CWinit'],
            SOGinit=self.params['SOGinit'], Sinit=self.params['Sinit'], t0=self.params['t0'],
            tcrit=self.params['tcrit'], tsumcrit=self.params['tsumcrit'],
            etmodel=inputs['etmodel'], LOGFLAG=inputs['LOGFLAG'], 
            NofDays=inputs['NofDays'], 
            day=inputs['DOY'],
            transp=torch.tensor([-999.], dtype=torch.float64), evap=torch.tensor([-999.], dtype=torch.float64),
            fWE=torch.tensor([-999.], dtype=torch.float64)
    )




# Prepare inputs
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data", "preprocessed_data", "")
data = pd.read_csv(''.join((data_path, 'station_21.csv')))
leng = 10
PAR = torch.tensor(data['PAR'][0:leng],dtype=torch.float64, requires_grad=True)
TAir = torch.tensor(data['Tair'][0:leng],dtype=torch.float64, requires_grad=True)
VPD = torch.tensor(data['VPD'][0:leng],dtype=torch.float64, requires_grad=True)
Precip = torch.tensor(data['Precip'][0:leng],dtype=torch.float64, requires_grad=True)
CO2 = torch.tensor(data['CO2'][0:leng], dtype=torch.float64, requires_grad=True)
fAPAR = torch.tensor(data['fapar'][0:leng],dtype=torch.float64, requires_grad=True)
DOY = torch.tensor(data['DOY'][0:leng], dtype=torch.float64, requires_grad=False)
GPP = torch.tensor(data['GPP'][0:leng], dtype=torch.float64, requires_grad=True)
ET = torch.tensor(data['ET'][0:leng], dtype=torch.float64, requires_grad=True)
NofDays = torch.tensor(1, dtype=torch.int)
logflag = 0
control = 0

# inputs = {
#     'PAR': PAR, 'TAir': TAir, 'VPD': VPD, 'Precip': Precip, 'CO2': CO2,
#     'fAPAR': fAPAR, 'DOY': DOY, 'GPP': GPP, 'ET': ET, 
#     'etmodel': control, 'LOGFLAG': logflag, 'NofDays': NofDays
# }

# Initialize placeholders
GPP_aggregated = []

# Initialize the model
preles_model = PrelesModel()

# Sequential processing using iloc
for t in range(len(data[0:leng])):
    row = data.iloc[t]  # Access the row for timestep `t`
    
    # Prepare inputs for the current timestep
    inputs = {
        'PAR': torch.tensor([row['PAR']], dtype=torch.float64, requires_grad=True),
        'TAir': torch.tensor([row['Tair']], dtype=torch.float64, requires_grad=True),
        'VPD': torch.tensor([row['VPD']], dtype=torch.float64, requires_grad=True),
        'Precip': torch.tensor([row['Precip']], dtype=torch.float64, requires_grad=True),
        'CO2': torch.tensor([row['CO2']], dtype=torch.float64, requires_grad=False),
        'fAPAR': torch.tensor([row['fapar']], dtype=torch.float64, requires_grad=True),
        'DOY': torch.tensor([row['DOY']], dtype=torch.float64, requires_grad=False),
        'GPP': torch.tensor([row['GPP']], dtype=torch.float64),  # Placeholder for GPP
        'ET': torch.tensor([row['ET']], dtype=torch.float64),   # Placeholder for ET
        'etmodel': control,
        'LOGFLAG': logflag,
        'NofDays': NofDays
    }

    # Forward pass
    outputs = preles_model.forward(**inputs)
    GPP_pred = outputs[0]  # Assuming first output corresponds to GPP

    # Collect GPP predictions
    GPP_aggregated.append(GPP_pred)

GPP_aggregated = torch.cat(GPP_aggregated, dim=0)
# Print aggregated results
print("Aggregated GPP Predictions:")
print(GPP_aggregated)
print(GPP)




# # Forward pass
# with torch.autograd.detect_anomaly():
#     outputs = preles_model.forward(**inputs)

#     print(outputs)


#     # Extract GPP output
#     GPP = outputs[0]
#     GPP.retain_grad()  # Retain gradient for non-leaf tensor

#     ET = outputs[1]
#     ET.retain_grad()  # Retain gradient for non-leaf tensor

#     # GPP.register_hook(hook_fn)
#     # ET.register_hook(hook_fn)

#     # Example backward pass
#     GPP_sum = GPP.sum()
#     ET_sum = ET.sum()

#     GPP_sum.backward(retain_graph=True)
#     ET_sum.backward(retain_graph=True)

#     # Print value and gradients for model parameters
#     for name, param in preles_model.params.items():
#         print(f"{name}: value {param.item()} gradient {param.grad}")

# # Print gradient for GPP
# # print("Gradient for GPP output:", GPP.grad)
# # print("Gradient for ET output:", ET.grad)


# # Gradient checking
# def gradcheck_preles():
#     # Use a simplified version of the forward pass for gradcheck
#     def forward_function(*params):
#         # Map tuple of inputs back to dictionary
#         param_dict = {key: param for key, param in zip(preles_model.params.keys(), params)}
#         preles_model.params.update(param_dict)
#         outputs = preles_model.forward(**inputs)
#         return outputs[0].sum() + outputs[1].sum()

#     # Extract all parameters for gradcheck
#     params = tuple(preles_model.params[key].detach().clone().requires_grad_() for key in preles_model.params)
    
#     return torch.autograd.gradcheck(forward_function, params, eps=1e-6, atol=1e-3)

# if gradcheck_preles():
#     print("Gradient check passed!")
# else:
#     print("Gradient check failed!")


# # # Attach hooks to parameters to inspect their gradients
# def parameter_hook(grad):
#     print("Gradient for parameter:", grad)

# for name, param in preles_model.params.items():
#     if param.requires_grad:
#         param.retain_grad()
#         param.register_hook(parameter_hook)

# # # Attach hooks to check intermediate gradients
# def hook_fn(grad):
#     print("Intermediate gradient:", grad)