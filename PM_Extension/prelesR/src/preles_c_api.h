// preles_c_api.h  (C header)
#ifndef PRELES_C_API_H
#define PRELES_C_API_H

#include "prelesglobals.h"  // <-- the *C* version where p1/p2/p3/p4 use double fields

#ifdef __cplusplus
extern "C" {
#endif

int preles(
    int NofDays,
    double *PAR, double *TAir, double *VPD, double *Precip, double *CO2,
    double *fAPAR,
    p1 Site_par, p2 GPP_par, p3 ET_par, p4 SnowRain_par,
    int etmodel,
    double *GPP, double *ET, double *SW, double *SOG,
    double *fS, double *fD, double *fW, double *fE,
    double *Throughfall, double *Interception, double *Snowmelt, double *Drainage,
    double *Canopywater,
    double *GPPmeas, double *ETmeas, double *SWmeas, double *S,
    int LOGFLAG, long int multisiteNday,
    int *day, double *transp, double *evap, double *fWE
);

#ifdef __cplusplus
}
#endif
#endif
