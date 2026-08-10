/*---------------------------------------------------------------------------*\
  Direct helium transport correlations used by the P418 pebble-bed study.
\*---------------------------------------------------------------------------*/

#ifndef hccbHeliumTransport_H
#define hccbHeliumTransport_H

#include "autoPtr.H"
#include "dictionary.H"

namespace Foam
{

template<class Thermo> class hccbHeliumTransport;

template<class Thermo>
inline hccbHeliumTransport<Thermo> operator+
(
    const hccbHeliumTransport<Thermo>&,
    const hccbHeliumTransport<Thermo>&
);

template<class Thermo>
inline hccbHeliumTransport<Thermo> operator*
(
    const scalar,
    const hccbHeliumTransport<Thermo>&
);

template<class Thermo>
Ostream& operator<<(Ostream&, const hccbHeliumTransport<Thermo>&);


template<class Thermo>
class hccbHeliumTransport
:
    public Thermo
{
    scalar viscosityCoefficient_;
    scalar viscosityTemperatureExponent_;
    scalar viscosityScale_;

    scalar conductivityCoefficient_;
    scalar referenceTemperature_;
    scalar conductivityTemperatureExponent_;
    scalar conductivityPressureCoefficient_;
    scalar conductivityPressureExponent_;
    scalar conductivityPressureTemperatureExponent_;
    scalar pressureScale_;

    scalar readCoeff(const word& coeffName, const dictionary& dict);


public:

    inline hccbHeliumTransport
    (
        const Thermo& t,
        const scalar viscosityCoefficient,
        const scalar viscosityTemperatureExponent,
        const scalar viscosityScale,
        const scalar conductivityCoefficient,
        const scalar referenceTemperature,
        const scalar conductivityTemperatureExponent,
        const scalar conductivityPressureCoefficient,
        const scalar conductivityPressureExponent,
        const scalar conductivityPressureTemperatureExponent,
        const scalar pressureScale
    );

    inline hccbHeliumTransport(const word&, const hccbHeliumTransport&);

    hccbHeliumTransport(const word& name, const dictionary& dict);

    inline autoPtr<hccbHeliumTransport> clone() const;

    static word typeName()
    {
        return "hccbHelium<" + Thermo::typeName() + '>';
    }

    inline scalar mu(const scalar p, const scalar T) const;

    inline scalar kappa(const scalar p, const scalar T) const;

    void write(Ostream& os) const;

    inline void operator+=(const hccbHeliumTransport&);

    inline void operator*=(const scalar);

    friend hccbHeliumTransport operator+ <Thermo>
    (
        const hccbHeliumTransport&,
        const hccbHeliumTransport&
    );

    friend hccbHeliumTransport operator* <Thermo>
    (
        const scalar,
        const hccbHeliumTransport&
    );

    friend Ostream& operator<< <Thermo>
    (
        Ostream&,
        const hccbHeliumTransport&
    );
};

} // End namespace Foam

#include "hccbHeliumTransportI.H"

#ifdef NoRepository
    #include "hccbHeliumTransport.C"
#endif

#endif

// ************************************************************************* //
