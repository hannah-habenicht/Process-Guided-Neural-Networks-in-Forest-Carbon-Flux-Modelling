#include "prelesglobals.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <stdio.h>

#include <iostream>

using namespace torch::indexing;

// Function declaration
int preles(
          int NofDays, torch::Tensor *PAR, torch::Tensor *TAir, torch::Tensor *VPD, torch::Tensor *Precip, 
          torch::Tensor *CO2, torch::Tensor *fAPAR,  p1 Site_par, p2 GPP_par, p3 ET_par,  p4 SnowRain_par, int etmodel,
          torch::Tensor *GPP, torch::Tensor *ET, torch::Tensor *SW, torch::Tensor *SOG,
          torch::Tensor *fS, torch::Tensor *fD, torch::Tensor *fW,  torch::Tensor *fE,
          torch::Tensor *Throughfall, torch::Tensor *Interception, torch::Tensor *Snowmelt,
          torch::Tensor *Drainage, torch::Tensor *Canopywater, torch::Tensor *GPPmeas, 
          torch::Tensor *ETmeas, torch::Tensor *SWmeas, torch::Tensor *S, 	   
          int LOGFLAG, long int multisiteNday, torch::Tensor *day, torch::Tensor *transp, 
          torch::Tensor *evap, torch::Tensor *fWE)
        {

        torch::Device device = PAR->device();
        auto options = PAR->options().device(device);

        extern torch::Tensor fS_model(torch::Tensor *S, torch::Tensor T, p2 GPP_par, torch::Tensor Msk, int i);

        extern torch::Tensor fPheno_model(
                p2 GPP_par, torch::Tensor T, torch::Tensor *PhenoS, 
                torch::Tensor DOY, torch::Tensor fS, torch::Tensor Msk, int i, torch::Tensor msk);

        extern torch::Tensor ETfun(
                torch::Tensor D, torch::Tensor theta, torch::Tensor ppfd, torch::Tensor fAPAR, 
                torch::Tensor T, p3 ET_par, p1 Site_par, torch::Tensor *canw, torch::Tensor *fE, torch::Tensor A,
                torch::Tensor fWgpp, p2 GPP_par,  //double fCO2mean, 
                torch::Tensor CO2, FILE *flog, int LOGFLAG, int etmodel, 
                torch::Tensor *transp, torch::Tensor *evap, torch::Tensor *fWE, int i, torch::Tensor Msk, torch::Tensor msk);
        
        extern void  interceptionfun(
                torch::Tensor *rain, torch::Tensor *intercepted, torch::Tensor Temp, p4
                SnowRain_par, torch::Tensor fAPAR, int i, torch::Tensor Msk, torch::Tensor msk);

        extern void swbalance(
                torch::Tensor *theta, torch::Tensor throughfall, 
                torch::Tensor snowmelt, torch::Tensor et, p1 sitepar, torch::Tensor *drainage,
                torch::Tensor *snow, torch::Tensor *canw, p4 SnowRain_par, int i, torch::Tensor Mask, torch::Tensor mask);

        extern void  Snow(
                torch::Tensor T, torch::Tensor *rain, torch::Tensor *snow, p4 SnowRain_par,
                torch::Tensor *SnowMelt, int i, torch::Tensor Msk, torch::Tensor msk);

        extern void initConditions(
                torch::Tensor **PAR, torch::Tensor **TAir, 
                torch::Tensor **VPD, torch::Tensor **Precip,
                torch::Tensor **CO2, torch::Tensor Msk, torch::Tensor msk);

        extern void GPPfun(
                torch::Tensor *gpp, torch::Tensor *gpp380, torch::Tensor ppfd,  
                torch::Tensor D, torch::Tensor CO2, torch::Tensor theta, 
                torch::Tensor fAPAR, torch::Tensor fSsub, 
                p2 GPP_par, p1 Site_par, torch::Tensor *fD, torch::Tensor *fW,
                torch::Tensor *fE, FILE *flog, int LOGFLAG, int i, torch::Tensor Msk, torch::Tensor msk);
        
      //extern double fCO2_VPD_exponent(double CO2, double xCO2 ) ;
      //extern double fCO2_ET_VPD_correction(double fEgpp, double xCO2 );
      //extern double fCO2_model_mean(double CO2, double bCO2 ) ;


        // Open log file
        FILE *flog=NULL;
        flog = fopen("preles.log", "a"); // EXCEPTION LOGGING
        if (LOGFLAG > 0.5) fprintf(flog, "  Stepped into preles()");

        // Initialize state variables
        torch::Tensor theta, theta_snow,theta_canopy, S_state; //soil_water, snow_storage, canopy_water, seasonal_state;
        torch::Tensor PhenoS = torch::tensor(0., options).requires_grad_();
        torch::Tensor fPheno = torch::tensor(0., options).requires_grad_();
        int i; //day_index
        torch::Tensor fEgpp = torch::tensor(0., options).requires_grad_();
        torch::Tensor gpp380 = torch::tensor(0., options).requires_grad_();
        
        // Masks for initialization
        int n = NofDays;
        torch::Tensor mask_day_start  = torch::zeros({n}, options);
        mask_day_start [0] = 1;
        mask_day_start.requires_grad_();
        torch::Tensor mask_day_rest  = torch::ones({n}, options);
        mask_day_rest [0] = 0;
        mask_day_rest .requires_grad_();
        
        // Initialize environmental variables
        initConditions(&PAR, &TAir, &VPD, &Precip, &CO2,mask_day_start, mask_day_rest );

        theta = torch::nansum(*SW*mask_day_start);
        theta_canopy = torch::nansum(*Canopywater*mask_day_start);
        theta_snow = torch::nansum(*SOG*mask_day_start);
        S_state = torch::nansum(*S*mask_day_start);
        

        if (LOGFLAG > 1.5) {
                        fprintf(flog, "   preles(): Starting values for storage components:\nSW=%f\tCW=%f\tSOG=%f\tS=%f\n",
                                theta.item<double>(), theta_canopy.item<double>(), theta_snow.item<double>(), S[0].item<double>());
                        fprintf(flog, "   ...will loop %d rows of weather input\n");
                        printf("   preles(): Site fAPAR =%f LUE =%f and soil depth =%f\n", 
                                fAPAR[0].item<double>(), GPP_par.beta.item<double>(), Site_par.soildepth.item<double>());
        }

        
        //    fclose(flog);
        
        
        /*------- LOOPING DAYS---------------------------------------------------*/
        /* ---------------- ----------------------------------------------------*/  
        //int d = NofDays.item<int>();

        // Main simulation loop
        for (i=0; i < n; i++) { 
                torch::Tensor mask = torch::ones({n}, options);
                mask[i] = 0;
                mask.requires_grad_();

                torch::Tensor Mask = torch::zeros({n}, options);
                Mask[i] = 1;
                Mask.requires_grad_();
                
                //Debug statement
                // std::cout << "Day " << i << ": Starting computation" << std::endl;

                if (LOGFLAG > 1.5) {
                        fprintf(flog, "   \ni=%d, NofDays=%d, SW=%f, CW=%f, SOG=%f, S=%f\n", 
                                i+1, NofDays, theta.item<double>(), theta_canopy.item<double>(), 
                                theta_snow.item<double>(), S_state.item<double>());
                }     

                // Log inputs
                // std::cout << "PAR: " << PAR[i] << ", TAir: " << TAir[i] << ", VPD: " << VPD[i] << ", Precip: " << Precip[i] << std::endl;

                
                /* Use previous day environment for prediction, if current values are missing,
                or suspicious*/
                if (i > 0) {                                //day_index 
                // Update previous day's values
                torch::Tensor bmmask = torch::zeros({n}, options);
                bmmask[i-1] = 1;
                bmmask.requires_grad_();
                
                torch::Tensor par = torch::zeros({n}, options);
                par[i] = torch::nansum(*PAR*bmmask);
                par.requires_grad_();

                torch::Tensor tair = torch::zeros({n}, options);
                tair[i] = torch::nansum(*TAir*bmmask);
                tair.requires_grad_();

                torch::Tensor vpd = torch::zeros({n}, options);
                vpd[i] = torch::nansum(*VPD*bmmask);
                vpd.requires_grad_();

                torch::Tensor precip = torch::zeros({n}, options);
                precip[i] = torch::nansum(*Precip*bmmask);
                precip.requires_grad_();

                torch::Tensor co2 = torch::zeros({n}, options);
                co2[i] = torch::nansum(*CO2*bmmask);
                co2.requires_grad_();

                torch::Tensor gppmeas = torch::zeros({n}, options);
                gppmeas[i] = torch::nansum(*GPPmeas*bmmask);
                gppmeas.requires_grad_();

                torch::Tensor etmeas = torch::zeros({n}, options);
                etmeas[i] = torch::nansum(*ETmeas*bmmask);
                etmeas.requires_grad_();

                torch::Tensor swmeas = torch::zeros({n}, options);
                swmeas[i] = torch::nansum(*SWmeas*bmmask);
                swmeas.requires_grad_();

                torch::Tensor sw = torch::zeros({n}, options);
                sw[i] = torch::nansum(*SW*bmmask);
                sw.requires_grad_();

                torch::Tensor sog = torch::zeros({n}, options);
                sog[i] = torch::nansum(*SOG*bmmask);
                sog.requires_grad_();
                
                if (torch::any(torch::lt(*PAR*Mask, -900)).item<bool>()) {
                        *PAR = *PAR * mask + par; 
                }
                if (torch::any(torch::lt(*TAir*Mask, -900)).item<bool>()) {
                        *TAir = *TAir * mask + tair;
                }
                if (torch::any(torch::lt(*VPD*Mask, 0)).item<bool>() || torch::any(torch::gt(*VPD*Mask, 6)).item<bool>())  {
                        *VPD = *VPD * mask + vpd;
                }
                if (torch::any(torch::lt(*Precip*Mask, 0)).item<bool>()) {
                        *Precip = *Precip * mask + (precip * 0.3); 
                }
                /* On avg. P+1=0.315*P 
                * (in Sodis & Hyde) */
                if (torch::any(torch::lt(*CO2*Mask, 0)).item<bool>()) {
                        *CO2 = *CO2 * mask + co2;
                }
                if (torch::any(torch::lt(*GPPmeas*Mask, -990)).item<bool>()) {
                        *GPPmeas = *GPPmeas * mask + gppmeas; 
                }
                if (torch::any(torch::lt(*ETmeas*Mask, -990)).item<bool>()) {
                        *ETmeas = *ETmeas * mask + etmeas;    
                } 
                if (torch::any(torch::lt(*SWmeas*Mask, 0.0)).item<bool>()) {
                        *SWmeas = *SWmeas * mask + swmeas;   
                }
                if (torch::any(torch::lt(*SW*Mask, -900)).item<bool>()) {
                        *SW = *SW * mask + sw; 
                }
                if (torch::any(torch::lt(*SOG*Mask, -900)).item<bool>()) {
                        *SOG = *SOG * mask + sog; // See above, could be used for 
                }
                
                }

                if ((LOGFLAG > 1.5)) {
                fprintf(flog, "   weather inputs: PAR=%f, T=%f, VPD=%f, P=%f, CO2=%f\n",
                        PAR[i].item<double>(), TAir[i].item<double>(), VPD[i].item<double>(), 
                        Precip[i].item<double>(), CO2[i].item<double>());
                }  

                // Phenology and photosynthesis models
                /* Update temperature state that tells us about seasonality -
                * for GPP and through GPP to ET */

                *fS = *fS * mask + fS_model(&S_state, *TAir, GPP_par, Mask, i);

                // Check for NaN or Inf
                if (torch::any(torch::isnan(*fS)).item<bool>()) {
                        std::cout << "NaN detected in fS at day " << i << std::endl;
                }

                torch::Tensor DAY = torch::nansum(*day*Mask);
                fPheno = fPheno_model(GPP_par, *TAir, &PhenoS, DAY, *fS, Mask, i, mask);
                *fAPAR = *fAPAR * mask + *fAPAR * Mask * fPheno; 

                // GPP function call
                GPPfun(&GPP[0], &gpp380, *PAR, *VPD, *CO2, theta, *fAPAR, *fS,
                        GPP_par, Site_par,  &fD[0], &fW[0], &fEgpp, 
                        flog, LOGFLAG, i, Mask, mask);
                
                /* Deciduous phenology - don't use if this information is inputted in fAPAR */
                /* Note also that fapar is multiplied by 0 or 1 (i.e. leaf development is not accounted for) */
                /* Model predicts budbreak based on critical threshold temperature sum */ 
                /* Note that this implementation works only if data starts before t0-date of fPheno-model */      
                
                // Update snow, interception, and soil water balances
                /* Calculate amount of snow and snowmelt at the end of the day */

                Snow(*TAir, &Precip[0], &theta_snow, SnowRain_par, &Snowmelt[0], i, Mask, mask); // NOTE: interception model could be better
                *Throughfall  = *Throughfall*mask + *Precip*Mask;
                interceptionfun(&Throughfall[0], &Interception[0], *TAir, SnowRain_par, *fAPAR, i, Mask, mask);
                

                /*Excess water from canopy will drip down to soil if not evaporated during the day, rest remains in canopy for the next day*/
                if (torch::lt(SnowRain_par.CWmax, 0.000000011).item<bool>()) { 
                        *Throughfall = *Throughfall + *Interception*Mask;
                } else {
                if (torch::any(torch::lt((SnowRain_par.CWmax* *fAPAR)*Mask, (*Interception + theta_canopy)*Mask)).item<bool>()) {
                        *Throughfall = *Throughfall + *Interception*Mask + theta_canopy*Mask - (SnowRain_par.CWmax* *fAPAR)*Mask;
                        theta_canopy = torch::nansum(SnowRain_par.CWmax  * *fAPAR*Mask);	
                } else {
                theta_canopy = torch::nansum(*Interception*Mask + theta_canopy*Mask);
                }     
                }
                

                *ET = *ET*mask + ETfun(*VPD, theta, *PAR, *fAPAR, *TAir, 
                                        ET_par, Site_par, &theta_canopy, &fE[0], // Soil water constrain on evaporation  
                                        gpp380, *fW,                             // soil water constrain of GPP at 380 ppm
                                        GPP_par,                                 //fCO2_ET_model_mean(CO2[i], GPP_par),
                                        *CO2, flog, LOGFLAG, etmodel, &transp[0], 
                                        &evap[0], &fWE[0], i, Mask, mask);
                
                
                /* Calculate soil water balance, drainage and related variables at the 
                end of the day, as well as update snow and canopy water with et */
                swbalance(&theta, *Throughfall, *Snowmelt, *ET,
                        Site_par, &Drainage[0], //&Psi[i], &Ks[i], 
                        &theta_snow, &theta_canopy, SnowRain_par, i, Mask, mask);
                

                /* Record result variables with storage components */
                *SOG = *SOG * mask + theta_snow * Mask;
                *SW = *SW * mask + theta * Mask;
                *Canopywater = *Canopywater * mask + theta_canopy * Mask;
                *S = *S*mask + S_state*Mask;

                if (LOGFLAG > 1.5) {
                fprintf(flog, "   preles(): estimated fS=%f\n", fS[i].item<double>());
                fprintf(flog, "   preles(): stepping into fPheno_model: inputs:\n GPP_par.t0=%f, T=%f, PhenoS=%f, DOY=%d\n", 
                        GPP_par.t0.item<double>(), TAir->item<double>(), PhenoS.item<double>(), day[i].item<int>());
                fprintf(flog, "   preles(): PhenoS=%f, fPheno=%f\n", 
                        PhenoS.item<double>(), fPheno.item<double>());
                fprintf(flog, "   preles(): fAPAR changed to %f\n", 
                        fAPAR[i].item<double>());
                fprintf(flog, "   preles(): GPP[i]=%f, fD[i]=%f, fEgpp=%f, gpp380=%f\n", 
                        GPP[i].item<double>(), fD[i].item<double>(), fEgpp.item<double>(), gpp380.item<double>());
                // fprintf(flog, "   preles(): estimated Thr.fall, Intercept, SOG, Snowmelt", 
                //         Throughfall->index({i}), Interception[i], SOG->index({i}), Snowmelt->index({i}));
                fprintf(flog, "   preles(): estimated canopy water=%f\n", 
                        theta_canopy.item<double>());
                fprintf(flog, "   preles(): ET=%lf\n", ET[i].item<double>());
                fprintf(flog, "   preles(): drainage, after ET: SW=%f \tSOG=%f CW=%f CW2=%f\n", 
                        Drainage[i].item<double>(), theta.item<double>(), theta_snow.item<double>(), theta_canopy.item<double>());
                fprintf(flog, "   preles(): after day state:\n   SW=%f\tCW=%f\tSOG=%f\tS=%f\n\n", 
                        SW[i].item<double>(), Canopywater[i].item<double>(), SOG[i].item<double>(), S[i].item<double>());
                }
                
        } // END DAY LOOP
        
        if (LOGFLAG > 1.5) {
                fprintf(flog, "   preles(): looped all days, closing preles.log, exiting...\n");
        }

        if (flog) {
                if (fclose(flog) != 0) {
                fprintf(stderr, "Error closing log file\n");
                return 1;
                }
        }

        return 0;
    }
        
