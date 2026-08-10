#include "rhoFluidThermo.H"
#include "pureMixture.H"
#include "specie.H"
#include "perfectGas.H"
#include "hConstThermo.H"
#include "sensibleEnthalpy.H"
#include "thermo.H"
#include "hccbHeliumTransport.H"
#include "makeFluidThermo.H"
#include "forThermo.H"

namespace Foam
{
    forThermo
    (
        hccbHeliumTransport,
        sensibleEnthalpy,
        hConstThermo,
        perfectGas,
        specie,
        makeFluidThermo,
        rhoFluidThermo,
        pureMixture
    );
}

// ************************************************************************* //
