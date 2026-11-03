// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

extern "C" {
  #include "preles_c_api.h"   // or the header that declares preles(...)
}

// Simple helper for 1D double arrays (C-contiguous)
using ND = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;
namespace py = pybind11;

static void check_len(const ND& a, ssize_t N, const char* name) {
    if (a.ndim() != 1) throw std::runtime_error(std::string(name) + " must be 1D");
    if (a.shape(0) != N) throw std::runtime_error(std::string(name) + " has length " +
        std::to_string(a.shape(0)) + ", expected " + std::to_string(N));
}

py::dict run_preles_numpy(
    // inputs (length N)
    ND PAR, ND TAir, ND VPD, ND Precip, ND CO2, ND fAPAR,
    ND GPPmeas, ND ETmeas, ND SWmeas,
    // day-of-year (length N; ints)
    py::array_t<int, py::array::c_style | py::array::forcecast> day,

    // parameters (scalars)
    double soildepth, double ThetaFC, double ThetaPWP, double tauDrainage,
    double beta, double tau, double S0, double Smax, double kappa, double gamma,
    double soilthres, double bCO2, double xCO2,
    double ETbeta, double ETkappa, double ETchi, double ETsoilthres, double ETnu,
    double MeltCoef, double I0, double CWmax, double SnowThreshold, double T_0,
    // initial states (scalars)
    double SWinit, double CWinit, double SOGinit, double Sinit,
    // phenology
    double t0, double tcrit, double tsumcrit,
    // controls
    int etmodel, int LOGFLAG
) {
    const ssize_t N = PAR.shape(0);
    // basic shape checks
    check_len(TAir, N, "TAir"); check_len(VPD, N, "VPD"); check_len(Precip, N, "Precip");
    check_len(CO2, N, "CO2"); check_len(fAPAR, N, "fAPAR");
    check_len(GPPmeas, N, "GPPmeas"); check_len(ETmeas, N, "ETmeas"); check_len(SWmeas, N, "SWmeas");
    if (day.ndim() != 1 || day.shape(0) != N) throw std::runtime_error("day must be 1D of length N");

    // Allocate outputs (length N)
    ND GPP   = ND(N);
    ND ET    = ND(N);
    ND SW    = ND(N);
    ND SOG   = ND(N);
    ND fS    = ND(N);
    ND fD    = ND(N);
    ND fW    = ND(N);
    ND fE    = ND(N);
    ND Throughfall  = ND(N);
    ND Interception = ND(N);
    ND Snowmelt     = ND(N);
    ND Drainage     = ND(N);
    ND Canopywater  = ND(N);
    ND S            = ND(N);
    ND transp       = ND(N);
    ND evap         = ND(N);
    ND fWE          = ND(N);

    // Set initial storages in the output arrays as in the C/R interface:
    // put initial values at index 0
    *GPP.mutable_data(0) = 0.0;
    *ET.mutable_data(0)  = 0.0;
    *SW.mutable_data(0)  = SWinit;
    *SOG.mutable_data(0) = SOGinit;
    *fS.mutable_data(0)  = 0.0;
    *fD.mutable_data(0)  = 0.0;
    *fW.mutable_data(0)  = 0.0;
    *fE.mutable_data(0)  = 0.0;
    *Throughfall.mutable_data(0)  = 0.0;
    *Interception.mutable_data(0) = 0.0;
    *Snowmelt.mutable_data(0)     = 0.0;
    *Drainage.mutable_data(0)     = 0.0;
    *Canopywater.mutable_data(0)  = CWinit;
    *S.mutable_data(0)            = Sinit;
    *transp.mutable_data(0)       = 0.0;
    *evap.mutable_data(0)         = 0.0;
    *fWE.mutable_data(0)          = 0.0;

    // Build parameter structs (C version: doubles, not tensors)
    p1 parSite; parSite.soildepth = soildepth; parSite.ThetaFC = ThetaFC;
    parSite.ThetaPWP = ThetaPWP;   parSite.tauDrainage = tauDrainage;

    p2 parGPP; parGPP.beta = beta; parGPP.tau = tau; parGPP.S0 = S0; parGPP.Smax = Smax;
    parGPP.kappa = kappa; parGPP.gamma = gamma; parGPP.soilthres = soilthres;
    parGPP.bCO2 = bCO2; parGPP.xCO2 = xCO2; parGPP.t0 = t0; parGPP.tcrit = tcrit; parGPP.tsumcrit = tsumcrit;

    p3 parET; parET.beta = ETbeta; parET.kappa = ETkappa; parET.chi = ETchi;
    parET.soilthres = ETsoilthres; parET.nu = ETnu;

    p4 parSnowRain; parSnowRain.MeltCoef = MeltCoef; parSnowRain.I0 = I0;
    parSnowRain.CWmax = CWmax; parSnowRain.SnowThreshold = SnowThreshold; parSnowRain.T_0 = T_0;

    // Call C model
    long int multisiteNday = N;  // matches R code pattern
    int rc = preles(
        static_cast<int>(N),
        PAR.mutable_data(), TAir.mutable_data(), VPD.mutable_data(), Precip.mutable_data(), CO2.mutable_data(),
        fAPAR.mutable_data(),
        parSite, parGPP, parET, parSnowRain, etmodel,
        GPP.mutable_data(), ET.mutable_data(), SW.mutable_data(), SOG.mutable_data(),
        fS.mutable_data(), fD.mutable_data(), fW.mutable_data(), fE.mutable_data(),
        Throughfall.mutable_data(), Interception.mutable_data(), Snowmelt.mutable_data(), Drainage.mutable_data(),
        Canopywater.mutable_data(),
        GPPmeas.mutable_data(), ETmeas.mutable_data(), SWmeas.mutable_data(), S.mutable_data(),
        LOGFLAG, multisiteNday,
        day.mutable_data(), transp.mutable_data(), evap.mutable_data(), fWE.mutable_data()
    );
    if (rc != 0) throw std::runtime_error("preles() returned non-zero code: " + std::to_string(rc));

    // Package results
    py::dict out;
    out["GPP"] = GPP; out["ET"] = ET; out["SW"] = SW; out["SOG"] = SOG;
    out["fS"] = fS; out["fD"] = fD; out["fW"] = fW; out["fE"] = fE;
    out["Throughfall"] = Throughfall; out["Interception"] = Interception; out["Snowmelt"] = Snowmelt; out["Drainage"] = Drainage;
    out["Canopywater"] = Canopywater; out["S"] = S;
    out["transp"] = transp; out["evap"] = evap; out["fWE"] = fWE;
    return out;
}

PYBIND11_MODULE(preles_c, m) {
    m.doc() = "Python wrapper for PRELES original C implementation (NumPy I/O)";
    m.def("run", &run_preles_numpy, "Run PRELES on 1D NumPy arrays");
}
