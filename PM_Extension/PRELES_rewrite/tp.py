import torch
import preles
import os
import math
import pandas as pd
import preles
from torch.autograd import grad, gradcheck
torch.autograd.set_detect_anomaly(True)
import ctypes

# Read in Data
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data", "preprocessed_data", "")
data = pd.read_csv(''.join((data_path, 'station_21.csv')))

def hook_fn(grad):
    print("Gradient value:", grad)
    return grad



leng = 10
gpp = torch.empty(leng)
et = torch.empty(leng)

PAR = torch.tensor(data['PAR'][0:leng], requires_grad=True)
TAir = torch.tensor(data['Tair'][0:leng], requires_grad=True)
VPD = torch.tensor(data['VPD'][0:leng], requires_grad=False)
Precip = torch.tensor(data['Precip'][0:leng], requires_grad=False)
CO2 = torch.tensor(data['CO2'][0:leng], dtype=torch.float64 ,requires_grad=False)
fAPAR = torch.tensor(data['fapar'][0:leng], requires_grad=False)
DOY = torch.tensor(data['DOY'][0:leng],dtype= torch.float64 ,requires_grad=False)
# print("PAR: ", PAR, "\nTAir: ", TAir, "\nVPD: ", VPD, "\nPrecip: ", Precip, "\nCO2: ", CO2, "\nfAPAR: ", fAPAR)

inputs = [PAR, TAir, VPD, Precip, CO2, fAPAR]

# Parameters acc. to import_preles.ipyn
p1 = torch.tensor(413., requires_grad=True)
p2 = torch.tensor(0.45, requires_grad=True)
p3 = torch.tensor(0.118, requires_grad=True)
p4 = torch.tensor(3., requires_grad=True)
p5 = torch.tensor(0.7457, requires_grad=True)
p6 = torch.tensor(10.93, requires_grad=True)
p7 = torch.tensor(-3.063, requires_grad=True)
p8 = torch.tensor(17.72, requires_grad=True)
p9 = torch.tensor(-0.1027, requires_grad=True)
p10 = torch.tensor(0.03673, requires_grad=True)
p11 = torch.tensor(0.7779, requires_grad=True)
p12 = torch.tensor(0.5, requires_grad=True)
p13 = torch.tensor(-0.364, requires_grad=True)
p14 = torch.tensor(0.2715, requires_grad=True)
p15 = torch.tensor(0.8351, requires_grad=True)
p16 = torch.tensor(0.07348, requires_grad=True)
p17 = torch.tensor(0.9996, requires_grad=True)
p18 = torch.tensor(0.4428, requires_grad=True)
p19 = torch.tensor(1.2, requires_grad=True)
p20 = torch.tensor(0.33, requires_grad=True)
p21 = torch.tensor(4.970496, requires_grad=True)
p22 = torch.tensor(0., requires_grad=True)
p23 = torch.tensor(0., requires_grad=True)
p24 = torch.tensor(160., requires_grad=True)
p25 = torch.tensor(0., requires_grad=True)
p26 = torch.tensor(0., requires_grad=True)
p27 = torch.tensor(0.0, requires_grad=False)
p28 = torch.tensor(-999., requires_grad=True)
p29 = torch.tensor(-999., requires_grad=True)
p30 = torch.tensor(-999., requires_grad=True)

GPPmeas = torch.tensor([-999.]*leng, requires_grad=False)
ETmeas = torch.tensor([-999.]*leng, requires_grad=False)
SWmeas = torch.tensor([-999.]*leng, requires_grad=False)
transp = torch.tensor([-999.]*leng, requires_grad=False)
evap = torch.tensor([-999.]*leng, requires_grad=False)
fWE = torch.tensor([-999.]*leng, requires_grad=False)
logflag = 0
control = 0
NofDays = torch.tensor(leng, dtype=torch.int)


    
# Outputs
GPP = torch.tensor([1e-6]*leng, dtype=torch.float64, requires_grad=True)
ET = torch.tensor([1e-6]*leng, dtype=torch.float64, requires_grad=True)
SW = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
SOG = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
fS = torch.tensor([0]*leng, dtype=torch.float64, requires_grad=False)
fD = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
fW = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
fE = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
Throughfall = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
Interception = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
Snowmelt = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
Drainage = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
Canopywater = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
S = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
# print("GPP: ", GPP, "\nET: ", ET, "\nSW: ", SW, "\nSOG: ", SOG, "\nfS: ", fS, "\nfD: ", fD, "\nfW: ", fW, "\nfE: ", 
#       fE, "\nThroughfall: ", Throughfall, "\nInterception: ", Interception, "\nSnowmelt: ", Snowmelt, "\nDrainage: ", 
#       Drainage, "\nCanopywater: ", Canopywater, "\nS: ", S)


# for i, inp in enumerate(inputs):
#     if torch.isnan(inp).any():
#         print(f"Input {i} contains NaN values: {inp}")
#     elif (inp < 0).any() and i != "CO2":
#         print(f"Input {i} contains negative values: {inp}")

# print("ET before backward pass:", ET)
# if torch.isnan(ET).any():
#     print("ET contains NaN values.")

inputs = (
    PAR.requires_grad_(),
    TAir.requires_grad_(),
    VPD.requires_grad_(),  # Convert if necessary
    Precip.requires_grad_(),
    CO2.requires_grad_(),
    fAPAR.requires_grad_(),
    GPPmeas.requires_grad_(),
    ETmeas.requires_grad_(),
    SWmeas.requires_grad_(),
    GPP.requires_grad_(),
    ET.requires_grad_(),
    SW.requires_grad_(),
    SOG.requires_grad_(),
    fS.requires_grad_(),
    fD.requires_grad_(),
    fW.requires_grad_(),
    fE.requires_grad_(),
    Throughfall.requires_grad_(),
    Interception.requires_grad_(),
    Snowmelt.requires_grad_(),
    Drainage.requires_grad_(),
    Canopywater.requires_grad_(),
    S.requires_grad_(),
    p1.requires_grad_(),  # soildepth
    p2.requires_grad_(),  # ThetaFC
    p3.requires_grad_(),  # ThetaPWP
    p4.requires_grad_(),  # tauDrainage
    p5.requires_grad_(),  # beta
    p6.requires_grad_(),  # tau
    p7.requires_grad_(),  # S0
    p8.requires_grad_(),  # Smax
    p9.requires_grad_(),  # kappa
    p10.requires_grad_(),  # gamma
    p11.requires_grad_(),  # soilthres
    p12.requires_grad_(),  # bCO2
    p13.requires_grad_(),  # xCO2
    p14.requires_grad_(),  # ETbeta
    p15.requires_grad_(),  # ETkappa
    p16.requires_grad_(),  # ETchi
    p17.requires_grad_(),  # ETsoilthres
    p18.requires_grad_(),  # ETnu
    p19.requires_grad_(),  # MeltCoef
    p20.requires_grad_(),  # I0
    p21.requires_grad_(),  # CWmax
    p22.requires_grad_(),  # SnowThreshold
    p23.requires_grad_(),  # T_0
    p24.requires_grad_(),  # SWinit
    p25.requires_grad_(),  # CWinit
    p26.requires_grad_(),  # SOGinit
    p27.requires_grad_(), # Sinit
    p28.requires_grad_(),  # t0
    p29.requires_grad_(),  # tcrit
    p30.requires_grad_(),  # tsumcrit
    torch.tensor(control, dtype=torch.float64, requires_grad=False),
    torch.tensor(logflag, dtype=torch.float64, requires_grad=False),
    torch.tensor(NofDays, dtype=torch.float64, requires_grad=False),
    DOY.requires_grad_(),
    transp.requires_grad_(),
    evap.requires_grad_(),
    fWE.requires_grad_()
)




print(' Run Preles ')
op = preles.preles(PAR=PAR, TAir = TAir, VPD = VPD, Precip = Precip, CO2 = CO2, fAPAR = fAPAR, GPPmeas = GPPmeas, ETmeas = ETmeas, 
                   SWmeas = SWmeas, GPP = GPP, ET = ET, SW = SW, SOG = SOG, fS = fS, fD = fD, fW = fW, fE = fE, Throughfall = Throughfall, 
                   Interception = Interception, Snowmelt = Snowmelt, Drainage = Drainage, Canopywater = Canopywater, S = S, soildepth = p1, 
                   ThetaFC = p2,  ThetaPWP = p3  , tauDrainage = p4  , beta = p5  , tau = p6  , S0= p7  , Smax = p8  , kappa= p9  , 
                   gamma = p10  , soilthres = p11  , bCO2 = p12  , xCO2 = p13  , ETbeta = p14  , ETkappa = p15  , ETchi = p16  , ETsoilthres = p17  , 
                   ETnu = p18  , MeltCoef = p19  , I0 = p20  , CWmax = p21  , SnowThreshold = p22  , T_0 = p23  , SWinit = p24  , CWinit = p25  ,
                   SOGinit = p26  , Sinit = p27  , t0 = p28  , tcrit = p29  , tsumcrit = p30  , etmodel = control  , LOGFLAG = logflag, 
                   NofDays = NofDays  , day = DOY  , transp = transp  , evap = evap  , fWE = fWE)

# Attach gradient hooks to outputs
# op[0].register_hook(hook_fn)

# # Compute and print gradients
# print("\nRunning Gradient Check:")
# test = gradcheck(preles.preles, inputs, eps=1e-6, atol=1e-4, rtol=1e-3, check_undefined_grad=True)
# print("Gradient check passed:", test)

# Output results
# print("\nOutputs from Preles:")
# for i, tensor in enumerate(op):
#     print(f"Output {i}: {tensor}")
#     if torch.isnan(tensor).any():
#         print(f"NaN detected in Output {i}")


# out_ET = ET.sum()

# for output in op:
#     if isinstance(output, torch.Tensor):
#         output.register_hook(hook_fn)

# Check gradients for the parameters
# parameters = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 
#               p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30]

# # # Check gradients
# for i, param in enumerate(parameters):
#     if param.grad is None:
#         print(f"Parameter p{i+1} has no gradient.")
#     else:
#         print(f"Parameter p{i+1} gradient shape: {param.grad.shape}, value: {param.grad}")

# print(p16.grad, p17.grad, p18.grad, p19.grad, p20.grad, p21.grad, p22.grad, p23.grad,p24.grad, p25.grad, p26.grad, p27.grad, p28.grad, p29.grad, p30.grad)
# print(grad(op, [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30], allow_unused=True))


GPP = op[0]
# # GPP.sum().backward(retain_graph=True)  # First backward pass
print(f"GPP", GPP)



# GPP_sum = GPP.sum()
# GPP_out = GPP_sum.backward(retain_graph=True)  # Backward pass
# print(GPP_out)


ET = op[1]
print(f"ET", ET)
# ET_sum = ET.sum()  # Create a scalar value
# ET_sum.backward()  # Backward pass


# print(ET_sum)



# #calls backward pass
# out_GPP.backward(retain_graph=True)
# print(p16.grad, p17.grad, p18.grad, p19.grad, p20.grad, p21.grad, p22.grad, p23.grad,p24.grad, p25.grad, p26.grad, p27.grad, p28.grad, p29.grad, p30.grad)
# print(grad(out, [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30], allow_unused=True))
# gpp[i] = GPP.clone().detach() 
# et[i] = ET.clone().detach()

# '''
# print(len(op[0]), max(op[0]), max(op[1]))
# print(op[0].backward())
# '''
# print(op[0].backward())# print(ET_sum)

# print(GPP.max())
# print(ET.max())
