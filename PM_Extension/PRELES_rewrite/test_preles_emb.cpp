
#include "prelesglobals.h"
#include <torch/torch.h>
#include <vector>
#include <iostream>

/* Parameter structure */
struct PrelesParameters {
    // Learnable parameters
    torch::Tensor soildepth;
    torch::Tensor ThetaFC;
    torch::Tensor ThetaPWP;
    torch::Tensor tauDrainage;
    torch::Tensor beta;
    torch::Tensor tau;
    torch::Tensor S0;
    torch::Tensor Smax;
    torch::Tensor kappa;
    torch::Tensor gamma;
    torch::Tensor soilthres;
    torch::Tensor bCO2;
    torch::Tensor xCO2;
    torch::Tensor ETbeta;
    torch::Tensor ETkappa;
    torch::Tensor ETchi;
    torch::Tensor ETsoilthres;
    torch::Tensor ETnu;
    torch::Tensor MeltCoef;
    torch::Tensor I0;
    torch::Tensor CWmax;
    torch::Tensor SnowThreshold;
    torch::Tensor T_0;
    torch::Tensor SWinit;
    torch::Tensor CWinit;
    torch::Tensor SOGinit;
    torch::Tensor Sinit;
    torch::Tensor t0;
    torch::Tensor tcrit;
    torch::Tensor tsumcrit;

    // State variables
    torch::Tensor GPP;
    torch::Tensor ET;
    torch::Tensor SW;
    torch::Tensor SOG;
    torch::Tensor fS;
    torch::Tensor fD;
    torch::Tensor fW;
    torch::Tensor fE;
    torch::Tensor Throughfall;
    torch::Tensor Interception;
    torch::Tensor Snowmelt;
    torch::Tensor Drainage;
    torch::Tensor Canopywater;
    torch::Tensor S;

    // Control arguments
    int LOGFLAG;
    int etmodel;
    int NofDays;
};



int main() {
    // Initialize parameters
    PrelesParameters params;
    params.soildepth = torch::tensor(1.0);
    params.ThetaFC = torch::tensor(0.45);

    // Print parameters to validate
    std::cout << "Soil Depth: " << params.soildepth << std::endl;
    std::cout << "ThetaFC: " << params.ThetaFC << std::endl;

    // Example tensor operation
    auto result = params.soildepth + params.ThetaFC;
    std::cout << "Result: " << result << std::endl;

    return 0;
}