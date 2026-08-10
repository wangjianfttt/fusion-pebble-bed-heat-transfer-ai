#include "argList.H"
#include "specie.H"
#include "perfectGas.H"
#include "hConstThermo.H"
#include "hccbHeliumTransport.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::noParallel();
    argList args(argc, argv);
    IOstream::defaultPrecision(16);
    Sout.precision(16);

    const specie helium("helium", 1.0, 3.992521804611004);
    const perfectGas<specie> eos(helium);
    const hConstThermo<perfectGas<specie>> thermo
    (
        eos,
        5200.0,
        0.0,
        Tstd,
        0.0
    );

    const hccbHeliumTransport<hConstThermo<perfectGas<specie>>> transport
    (
        thermo,
        0.4646,
        0.66,
        1e-6,
        0.1448,
        273.0,
        0.68,
        2.5e-3,
        1.17,
        -1.85,
        1e6
    );

    const scalar pressures[] = {93413.29837, 120000.0, 145976.4733};
    const scalar temperatures[] = {299.0, 500.0, 700.0, 1001.0};

    Sout<< "p_pa,T_k,mu_pa_s,kappa_w_m_k" << nl;

    for (const scalar p : pressures)
    {
        for (const scalar T : temperatures)
        {
            Sout<< p << ',' << T << ',' << transport.mu(p, T) << ','
                << transport.kappa(p, T) << nl;
        }
    }

    return 0;
}

// ************************************************************************* //
