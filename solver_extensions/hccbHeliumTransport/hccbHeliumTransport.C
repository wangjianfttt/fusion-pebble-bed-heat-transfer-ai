/*---------------------------------------------------------------------------*\
  Dictionary construction and output for the direct helium transport model.
\*---------------------------------------------------------------------------*/

#include "hccbHeliumTransport.H"
#include "IOstreams.H"

template<class Thermo>
Foam::scalar Foam::hccbHeliumTransport<Thermo>::readCoeff
(
    const word& coeffName,
    const dictionary& dict
)
{
    return dict.subDict("transport").lookup<scalar>(coeffName);
}


template<class Thermo>
Foam::hccbHeliumTransport<Thermo>::hccbHeliumTransport
(
    const word& name,
    const dictionary& dict
)
:
    Thermo(name, dict),
    viscosityCoefficient_(readCoeff("viscosityCoefficient", dict)),
    viscosityTemperatureExponent_
    (
        readCoeff("viscosityTemperatureExponent", dict)
    ),
    viscosityScale_(readCoeff("viscosityScale", dict)),
    conductivityCoefficient_(readCoeff("conductivityCoefficient", dict)),
    referenceTemperature_(readCoeff("referenceTemperature", dict)),
    conductivityTemperatureExponent_
    (
        readCoeff("conductivityTemperatureExponent", dict)
    ),
    conductivityPressureCoefficient_
    (
        readCoeff("conductivityPressureCoefficient", dict)
    ),
    conductivityPressureExponent_
    (
        readCoeff("conductivityPressureExponent", dict)
    ),
    conductivityPressureTemperatureExponent_
    (
        readCoeff("conductivityPressureTemperatureExponent", dict)
    ),
    pressureScale_(readCoeff("pressureScale", dict))
{}


template<class Thermo>
void Foam::hccbHeliumTransport<Thermo>::write(Ostream& os) const
{
    os  << this->name() << endl
        << token::BEGIN_BLOCK << incrIndent << nl;

    Thermo::write(os);

    dictionary dict("transport");
    dict.add("viscosityCoefficient", viscosityCoefficient_);
    dict.add
    (
        "viscosityTemperatureExponent",
        viscosityTemperatureExponent_
    );
    dict.add("viscosityScale", viscosityScale_);
    dict.add("conductivityCoefficient", conductivityCoefficient_);
    dict.add("referenceTemperature", referenceTemperature_);
    dict.add
    (
        "conductivityTemperatureExponent",
        conductivityTemperatureExponent_
    );
    dict.add
    (
        "conductivityPressureCoefficient",
        conductivityPressureCoefficient_
    );
    dict.add
    (
        "conductivityPressureExponent",
        conductivityPressureExponent_
    );
    dict.add
    (
        "conductivityPressureTemperatureExponent",
        conductivityPressureTemperatureExponent_
    );
    dict.add("pressureScale", pressureScale_);

    os  << indent << dict.dictName() << dict
        << decrIndent << token::END_BLOCK << nl;
}


template<class Thermo>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const hccbHeliumTransport<Thermo>& ht
)
{
    ht.write(os);
    return os;
}

// ************************************************************************* //
