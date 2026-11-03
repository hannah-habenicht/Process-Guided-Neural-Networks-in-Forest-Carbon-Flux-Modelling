import os
import torch
import torch.nn as nn
import preles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


# Define the Preles model with manual gradient tracking
class PrelesModel:
    def __init__(self):
        self.device = None
        
        self.params = {
            'soildepth': torch.tensor(413., requires_grad=True),
            'ThetaFC': torch.tensor(0.45, requires_grad=False),
            'ThetaPWP': torch.tensor(0.118, requires_grad=False),
            'tauDrainage': torch.tensor(3., requires_grad=False),
            'beta': torch.tensor(0.7457, requires_grad=True),
            'tau': torch.tensor(10.93, requires_grad=True),
            'S0': torch.tensor(-3.063, requires_grad=True),
            'Smax': torch.tensor(17.72, requires_grad=True),
            'kappa': torch.tensor(-0.1027, requires_grad=True),
            'gamma': torch.tensor(0.03673, requires_grad=True),
            'soilthres': torch.tensor(0.7779, requires_grad=True),
            'bCO2': torch.tensor(0.5, requires_grad=False),
            'xCO2': torch.tensor(-0.364, requires_grad=False),
            'ETbeta': torch.tensor(0.2715, requires_grad=True),
            'ETkappa': torch.tensor(0.8351, requires_grad=True),
            'ETchi': torch.tensor(0.07348, requires_grad=True),
            'ETsoilthres': torch.tensor(0.9996, requires_grad=True),
            'ETnu': torch.tensor(0.4428, requires_grad=True),
            'MeltCoef': torch.tensor(1.2, requires_grad=False),
            'I0': torch.tensor(0.33, requires_grad=False),
            'CWmax': torch.tensor(4.970496, requires_grad=False),
            'SnowThreshold': torch.tensor(0., requires_grad=False),
            'T_0': torch.tensor(0., requires_grad=False),
            'SWinit': torch.tensor(160., requires_grad=False),
            'CWinit': torch.tensor(0., requires_grad=False),
            'SOGinit': torch.tensor(0., requires_grad=False),
            'Sinit': torch.tensor(0.0, requires_grad=False),  # Optional: Not trainable
            't0': torch.tensor(-999., requires_grad=False),
            'tcrit': torch.tensor(-999., requires_grad=False),
            'tsumcrit': torch.tensor(-999., requires_grad=False),}
        

    def forward(self, inputs):
        # Explicitly map parameters to preles arguments
        #another working option
        test = dict( PAR=inputs['PAR'], TAir=inputs['TAir'], VPD=inputs['VPD'], Precip=inputs['Precip'], CO2=inputs['CO2'],
            fAPAR=inputs['fAPAR'], GPPmeas=inputs.get('GPPmeas', torch.tensor([-999.], dtype=torch.float64)),
            ETmeas=inputs.get('ETmeas', torch.tensor([-999.], dtype=torch.float64)),
            SWmeas=inputs.get('SWmeas', torch.tensor([-999.], dtype=torch.float64)),
            GPP=inputs['GPP'], ET=inputs['ET'], SW=torch.tensor([0.], dtype=torch.float64),
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
            etmodel=inputs['etmodel'], LOGFLAG=inputs['LOGFLAG'], NofDays=inputs['NofDays'], day=inputs['DOY'],
            transp=torch.tensor([-999.], dtype=torch.float64), evap=torch.tensor([-999.], dtype=torch.float64),
            fWE=torch.tensor([-999.], dtype=torch.float64))

        return preles.preles(*list(test.values()))

# Load and preprocess the input data
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data", "preprocessed_data", "")
data = pd.read_csv(''.join((data_path, 'station_21.csv')))

leng = len(data[0:2])  # Adjust input length as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAR = torch.tensor(data['PAR'][0:leng], dtype=torch.float64, requires_grad=True)
TAir = torch.tensor(data['Tair'][0:leng], dtype=torch.float64, requires_grad=True)
VPD = torch.tensor(data['VPD'][0:leng], dtype=torch.float64, requires_grad=True)
Precip = torch.tensor(data['Precip'][0:leng], dtype=torch.float64, requires_grad=True)
CO2 = torch.tensor(data['CO2'][0:leng], dtype=torch.float64, requires_grad=True)
CO2 = torch.tensor(data['CO2'][0:leng], dtype=torch.float64, requires_grad=True)
fAPAR = torch.tensor(data['fapar'][0:leng], dtype=torch.float64, requires_grad=True)
DOY = torch.tensor(data['DOY'][0:leng], dtype=torch.float64, requires_grad=True)
DOY = torch.tensor(data['DOY'][0:leng], dtype=torch.float64, requires_grad=True)
GPP_obs = torch.tensor(data['GPP'][0:leng], dtype=torch.float64, requires_grad=True)
ET_obs = torch.tensor(data['ET'][0:leng], dtype=torch.float64, requires_grad=True)

# Initialize model and optimizer
preles_model = PrelesModel()
optimizer = torch.optim.Adam(preles_model.params.values(), lr=1000)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 100
loss_hist= list()
param_hist = preles_model.params.copy()
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients

    # Forward pass
    inputs = {
        'PAR': PAR, 'TAir': TAir, 'VPD': VPD, 'Precip': Precip, 'CO2': CO2,
        'fAPAR': fAPAR, 'DOY': DOY, 'GPP': torch.zeros_like(GPP_obs), 'ET': torch.zeros_like(ET_obs),
        'etmodel': torch.tensor(0, dtype=torch.float64, device=device), 
        'LOGFLAG': torch.tensor(0, dtype=torch.float64, device=device),  # Changed to tensor
        'NofDays': torch.tensor(leng, dtype=torch.float64, device=device)  # Ensure it is a tensor
    }
    
    # Check for NaN or Inf in inputs
    for name, tensor in inputs.items():
        if torch.isnan(tensor).any():
            print(f"NaN found in input: {name}")
        if torch.isinf(tensor).any():
            print(f"Inf found in input: {name}")
    
    
    
    #test= *list(inputs.values())
    outputs = preles_model.forward(inputs)
    GPP_pred, ET_pred = outputs[0], outputs[1]

    # Check for NaN or Inf in outputs
    if torch.isnan(GPP_pred).any() or torch.isnan(ET_pred).any():
        print("NaN detected in outputs")
    if torch.isinf(GPP_pred).any() or torch.isinf(ET_pred).any():
        print("Inf detected in outputs")
    
    # Compute loss, check outputs and MSE definition
    #loss = loss_fn(GPP_pred, GPP_obs) + loss_fn(ET_pred, ET_obs) #mean missing I think
    #other option
    obs = torch.stack((GPP_obs,ET_obs))
    out = torch.stack((outputs[0],outputs[1]))
    loss= loss_fn(out,obs)
    # Add L2 regularization, what are doing there ? 
    # l2_lambda = 0.01
    # l2_reg = sum((param ** 2).sum() for param in preles_model.params.values())
    # loss += l2_lambda * l2_reg

    # Backward pass
    loss.backward()
    loss_hist.append(loss.detach().numpy())

    #outputs[epoch].backward()

# Apply gradient clipping
# torch.nn.utils.clip_grad_norm_(preles_model.params.values(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    # Log progress
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Final parameter values
    # for param in preles_model.params.parameters():
        
    #     print(f"{name}: {param.item()}")
    for name,i in inputs.items():
        if i.grad!=None:
            print(epoch)
            print(epoch,name,i.grad)

    # Final parameter values
    # for param in preles_model.params.parameters():
        
    #     print(f"{name}: {param.item()}")
    for name,i in inputs.items():
        if i.grad!=None:
            print(epoch)
            print(epoch,name,i.grad)

    for name, param in preles_model.params.items():
        if param.grad is not None:
            print(f"{name} gradient:{param.grad}\n")
            print(f"{name} value:{param}\n")

            if isinstance(param_hist[name],List):
                param_hist[name].append(param.clone().detach().numpy())
            else:
                h= [param_hist[name].clone().detach().numpy()]
                param_hist[name]= h
                param_hist[name].append(param.clone().detach().numpy())

    
    #     # else:
        #     print(f"{name} has no gradient (likely not used in forward pass)")

plt.figure()
plt.plot(np.arange(len(loss_hist)), loss_hist)
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.show()

# plt.figure()
# plt.plot(np.arange(len(param_hist['beta'])), param_hist['ETkappa'])
# plt.xlabel('epochs')
# plt.ylabel('values')
# plt.show()

print('blub')