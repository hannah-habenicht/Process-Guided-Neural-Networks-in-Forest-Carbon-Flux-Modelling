#include "prelesglobals.h"
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <stdio.h>

/* Seasonality model of Mäkelä et al 2004 */
torch::Tensor fS_model(torch::Tensor *S, torch::Tensor T, p2 GPP_par, torch::Tensor Msk, int i) {
  
  auto device = S->device();
  auto options = S->options().device(device);

  
  torch::Tensor fS;
  
  if (i>0){
    torch::Tensor smsk = torch::zeros(Msk.sizes(), options);
    smsk[i-1] = 1;
    smsk.requires_grad_();
    *S = torch::nansum(*S*smsk);
    //*S = torch::tensor(-0.85436861, torch::requires_grad());
  }
  
  // Core update
  *S = *S + ((T-*S).pow(Msk)/GPP_par.tau.pow(Msk))*Msk;
  
  if (torch::any(torch::lt((*S*Msk-GPP_par.S0*Msk), 0)).item<bool>()){
    fS = Msk * torch::tensor(0., options).requires_grad_();
  } else {
    fS = (*S*Msk-GPP_par.S0*Msk);
  }
  if (torch::any(torch::gt((fS.pow(Msk)/GPP_par.Smax.pow(Msk))*Msk, 1*Msk)).item<bool>()){
    fS = Msk* torch::tensor(1., options).requires_grad_();
  } else {
    fS = (fS.pow(Msk)/GPP_par.Smax.pow(Msk))*Msk;
  }
  
  
  return(fS);
};


torch::Tensor fPheno_model(p2 GPP_par, torch::Tensor T, torch::Tensor *PhenoS, 
			    torch::Tensor DOY, torch::Tensor fS, torch::Tensor Msk, int i, torch::Tensor msk) {
          torch::Tensor m; 
          auto options = msk.options().device(msk.device());
          torch::Tensor fPheno=torch::zeros(msk.sizes(), options).requires_grad_();
  
  if (torch::gt(GPP_par.t0, -998).item<bool>()) { // ie not -999 
    
    /* Budbreak must occur between specified min. date and end of July */
    if (torch::gt(DOY, (GPP_par.t0 - 0.5)).item<bool>() && torch::lt(DOY, 213).item<bool>() )  {
      m = (T*Msk - GPP_par.tcrit*Msk);

      if (torch::any(torch::lt(m*Msk, 0)).item<bool>()){
        m = m * msk; //msk will be 0 everywhere except i, therefore  so any(...) really just checks m[i] < 0.
        }

      *PhenoS = *PhenoS + m;
    
    } else {
      *PhenoS = *PhenoS * msk;
    }
    
    if (torch::any(torch::gt(*PhenoS*Msk, (GPP_par.tsumcrit - 0.005)*Msk)).item<bool>()){
      fPheno = fPheno + Msk;
    } else {
      fPheno = fPheno * msk;
    }
    /* Quick solution to leaf out: 
     * After end of July we just apply season prediction based on conifer fS 
     *  for gradual leaf out. Assume leaves drop much faster that fS.
     *  ...essentially this should be light driven process...i think. */
    if (torch::gt(DOY, 212).item<bool>()) {
        std::cout << "-fPh7-";
        fPheno = fPheno*msk + (fS*Msk) * (fS*Msk);

        if (torch::any(torch::lt(fPheno*Msk, 0.5*Msk)).item<bool>()) {
            std::cout << "-fPh8-";
            fPheno = fPheno * msk;
        } 
    }
    
    /* If there is no t0 parameter, it is an evergreen */
  } else {
      fPheno = fPheno*msk + Msk;
  }
  
  return(fPheno);
};

/* *****************************************************************/
/* f-modifiers for increasing CO2 prepared by P. Kolari, pers. comm.*/
/*double fCO2_model_mean(double CO2, p2 GPP_par ) {
  return(1 + (CO2-380)/(CO2-380+GPP_par.bCO2));
}
double fCO2_VPD_exponent(double CO2, p2 GPP_par ) {
  return(pow(CO2/380, GPP_par.xCO2));
}
*/
/*
double fCO2_model_mean(double CO2, double b ) {
  return(1 + (CO2-380)/(CO2-380+b));
}
double fCO2_VPD_exponent(double CO2, double xCO2 ) {
  return(pow(380/CO2, xCO2));
}
*/

/* Note: 'ET_par.bC02' is the same as GPP_par.bCO2 */
/*
double fCO2_ET_model_mean(double CO2, p2 GPP_par ) {
  return(1 - 1.95*(CO2-380)/(CO2-380+(GPP_par.bCO2)));
}
*/
/* *****************************************************************/
/* New CO2 modifiers based on APES simulator (Launiainen et al.)
   which account for the energy balance of the forest. Fitted responses
   to model predicted
   bCO2 = 0.5; xCO2 = -0.364
*/
torch::Tensor fCO2_model_mean(torch::Tensor CO2, p2 GPP_par, torch::Tensor Msk) {
  return(Msk + GPP_par.bCO2 * torch::log(CO2/380) * Msk);
}
torch::Tensor fCO2_ET_model_mean(torch::Tensor CO2, p2 GPP_par, torch::Tensor Msk ) {
  return(Msk + GPP_par.xCO2 * torch::log(CO2/380) * Msk);
}



/* GPP model, modified from Mäkelä et al 2008 */
void GPPfun(torch::Tensor *gpp, torch::Tensor *gpp380, 
	    torch::Tensor ppfd,  torch::Tensor D, torch::Tensor CO2, torch::Tensor theta, 
	    torch::Tensor fAPAR, torch::Tensor fSsub, 
	    p2 GPP_par, p1 Site_par, torch::Tensor *fD, torch::Tensor *fW,
	    torch::Tensor *fE, FILE *flog, int LOGFLAG, int i, torch::Tensor Msk, torch::Tensor msk) {

  extern torch::Tensor fCO2_model_mean(torch::Tensor CO2, p2 b, torch::Tensor Msk) ;
  //    extern double fCO2_VPD_exponent(double CO2, double xCO2 ) ;
  torch::Tensor thetavol = (theta/Site_par.soildepth);

  //  double GPPsub, GPP380sub;
  torch::Tensor fCO2;
  torch::Tensor REW=(thetavol-Site_par.ThetaPWP)/
        (Site_par.ThetaFC-Site_par.ThetaPWP);

  auto options = ppfd.options().device(ppfd.device());

  torch::Tensor fEsub;
  torch::Tensor fWsub;
  torch::Tensor fLsub;
  torch::Tensor fDsub;
  // double fECO2sub, fDCO2sub, fWCO2sub;
  /* first the reference condition (ca=380 ppm) effect */
  fDsub = torch::exp(GPP_par.kappa * D)*Msk;                    //kappa is a tunable parameter
  if (torch::any(torch::gt(fDsub*Msk, 1*Msk)).item<bool>()){
    
    fDsub = fDsub * msk + Msk;
  }
 
  if (torch::lt(GPP_par.soilthres, -998).item<bool>()) { 
    fWsub = Msk;      /* e.g. -999 means no water control of GPP*/
  } else {
    
    if (torch::any(torch::lt(REW*Msk, GPP_par.soilthres*Msk)).item<bool>()){ 
      if (torch::any(torch::gt(REW*Msk, 0.01*Msk)).item<bool>()){
	fWsub = (REW/GPP_par.soilthres)*Msk;
      } else {
	fWsub = torch::zeros(Msk.sizes(), options).requires_grad_();
      }
    } else {
	fWsub = Msk;
      }
  }
  
  fLsub = (torch::ones(Msk.sizes(),options).requires_grad_() / (Msk*GPP_par.gamma*ppfd + torch::ones(Msk.sizes(), options).requires_grad_()))*Msk;
  
  if (torch::any(torch::gt(fDsub*Msk, fWsub*Msk)).item<bool>()){
    fEsub = fWsub;
  } else {
    fEsub = fDsub;
  }
  
  *fW = *fW*msk + fWsub;
  *fD = *fD*msk + fEsub;
  *gpp380 = *gpp380*msk + (GPP_par.beta *  ppfd *  fAPAR * fSsub * fLsub * fEsub)*Msk;
  fCO2 = fCO2_model_mean(CO2, GPP_par, Msk);
  *gpp = *gpp*msk + (*gpp380 * fCO2)*Msk;
    
  if (LOGFLAG > 1.5) 
    fprintf(flog, 
      "   gpp(): Modifiers: fAPAR %f\tfSsub %f\t fLsub %f\t fDsub %f\t fWsub %f\tfEsub %f\t fCO2 %f\n                    gpp380 %f\t gpp %f\n",
      fAPAR.item<double>(), fSsub.item<double>(), fLsub.item<double>(), fDsub.item<double>(), fWsub.item<double>(), fEsub.item<double>(), fCO2.item<double>(), gpp380->item<double>(), gpp->item<double>());



    /* This has been removed, and simpler multiplicative CO2 modifier to gpp380 is used.
    * CO2 effect not only influences fD but also fW, due to stomatal action
    
    fDCO2sub = fDsub * pow(exp(-0.4 * D),
			   fCO2_VPD_exponent(CO2, GPP_par.xCO2)) / exp(-0.4 * D) ;
    fWCO2sub = fWsub * pow(fWsub, fCO2_VPD_exponent(CO2, GPP_par.xCO2));

    if (LOGFLAG > 1.5) 
      fprintf(flog, 
	      "   gpp(): Modifier values for GPP at %lf\n      fD=%lf\tfW=%lf\tfCO2mean=%lf\n", 
	      CO2, fDCO2sub, fWCO2sub, fCO2_model_mean(CO2, GPP_par.bCO2));
    
    if (fDCO2sub > fWCO2sub) fECO2sub=fWCO2sub; else fECO2sub = fDCO2sub;
    
    *fECO2 = fECO2sub;
    
    *gpp = GPP_par.beta *  ppfd *  fAPAR * fSsub * fLsub  * fECO2sub * 
      fCO2_model_mean(CO2, GPP_par.bCO2);
*/

}
