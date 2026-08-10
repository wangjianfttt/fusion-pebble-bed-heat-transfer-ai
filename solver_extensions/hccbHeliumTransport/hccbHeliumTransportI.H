/*---------------------------------------------------------------------------*\
  Inline implementation of the direct helium transport correlations.
\*---------------------------------------------------------------------------*/

template<class Thermo>
inline Foam::hccbHeliumTransport<Thermo>::hccbHeliumTransport
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
)
:
    Thermo(t),
    viscosityCoefficient_(viscosityCoefficient),
    viscosityTemperatureExponent_(viscosityTemperatureExponent),
    viscosityScale_(viscosityScale),
    conductivityCoefficient_(conductivityCoefficient),
    referenceTemperature_(referenceTemperature),
    conductivityTemperatureExponent_(conductivityTemperatureExponent),
    conductivityPressureCoefficient_(conductivityPressureCoefficient),
    conductivityPressureExponent_(conductivityPressureExponent),
    conductivityPressureTemperatureExponent_
    (
        conductivityPressureTemperatureExponent
    ),
    pressureScale_(pressureScale)
{}


template<class Thermo>
inline Foam::hccbHeliumTransport<Thermo>::hccbHeliumTransport
(
    const word& name,
    const hccbHeliumTransport& ht
)
:
    Thermo(name, ht),
    viscosityCoefficient_(ht.viscosityCoefficient_),
    viscosityTemperatureExponent_(ht.viscosityTemperatureExponent_),
    viscosityScale_(ht.viscosityScale_),
    conductivityCoefficient_(ht.conductivityCoefficient_),
    referenceTemperature_(ht.referenceTemperature_),
    conductivityTemperatureExponent_(ht.conductivityTemperatureExponent_),
    conductivityPressureCoefficient_(ht.conductivityPressureCoefficient_),
    conductivityPressureExponent_(ht.conductivityPressureExponent_),
    conductivityPressureTemperatureExponent_
    (
        ht.conductivityPressureTemperatureExponent_
    ),
    pressureScale_(ht.pressureScale_)
{}


template<class Thermo>
inline Foam::autoPtr<Foam::hccbHeliumTransport<Thermo>>
Foam::hccbHeliumTransport<Thermo>::clone() const
{
    return autoPtr<hccbHeliumTransport<Thermo>>
    (
        new hccbHeliumTransport<Thermo>(*this)
    );
}


template<class Thermo>
inline Foam::scalar Foam::hccbHeliumTransport<Thermo>::mu
(
    const scalar p,
    const scalar T
) const
{
    if (p <= 0 || T <= 0)
    {
        FatalErrorInFunction
            << "Helium pressure and temperature must be positive; p="
            << p << " Pa, T=" << T << " K" << exit(FatalError);
    }

    return
        viscosityCoefficient_
       *::pow(T, viscosityTemperatureExponent_)
       *viscosityScale_;
}


template<class Thermo>
inline Foam::scalar Foam::hccbHeliumTransport<Thermo>::kappa
(
    const scalar p,
    const scalar T
) const
{
    if (p <= 0 || T <= 0)
    {
        FatalErrorInFunction
            << "Helium pressure and temperature must be positive; p="
            << p << " Pa, T=" << T << " K" << exit(FatalError);
    }

    const scalar reducedT = T/referenceTemperature_;
    const scalar reducedP = p/pressureScale_;

    return
        conductivityCoefficient_
       *::pow(reducedT, conductivityTemperatureExponent_)
       *(
            1
          + conductivityPressureCoefficient_
           *::pow(reducedP, conductivityPressureExponent_)
           *::pow(reducedT, conductivityPressureTemperatureExponent_)
        );
}


template<class Thermo>
inline void Foam::hccbHeliumTransport<Thermo>::operator+=
(
    const hccbHeliumTransport& ht
)
{
    scalar Y1 = this->Y();
    Thermo::operator+=(ht);

    if (mag(this->Y()) > small)
    {
        Y1 /= this->Y();
        const scalar Y2 = ht.Y()/this->Y();

        viscosityCoefficient_ =
            Y1*viscosityCoefficient_ + Y2*ht.viscosityCoefficient_;
        viscosityTemperatureExponent_ =
            Y1*viscosityTemperatureExponent_
          + Y2*ht.viscosityTemperatureExponent_;
        viscosityScale_ = Y1*viscosityScale_ + Y2*ht.viscosityScale_;
        conductivityCoefficient_ =
            Y1*conductivityCoefficient_ + Y2*ht.conductivityCoefficient_;
        referenceTemperature_ =
            Y1*referenceTemperature_ + Y2*ht.referenceTemperature_;
        conductivityTemperatureExponent_ =
            Y1*conductivityTemperatureExponent_
          + Y2*ht.conductivityTemperatureExponent_;
        conductivityPressureCoefficient_ =
            Y1*conductivityPressureCoefficient_
          + Y2*ht.conductivityPressureCoefficient_;
        conductivityPressureExponent_ =
            Y1*conductivityPressureExponent_
          + Y2*ht.conductivityPressureExponent_;
        conductivityPressureTemperatureExponent_ =
            Y1*conductivityPressureTemperatureExponent_
          + Y2*ht.conductivityPressureTemperatureExponent_;
        pressureScale_ = Y1*pressureScale_ + Y2*ht.pressureScale_;
    }
}


template<class Thermo>
inline void Foam::hccbHeliumTransport<Thermo>::operator*=(const scalar s)
{
    Thermo::operator*=(s);
}


template<class Thermo>
inline Foam::hccbHeliumTransport<Thermo> Foam::operator+
(
    const hccbHeliumTransport<Thermo>& ht1,
    const hccbHeliumTransport<Thermo>& ht2
)
{
    Thermo t
    (
        static_cast<const Thermo&>(ht1) + static_cast<const Thermo&>(ht2)
    );

    if (mag(t.Y()) < small)
    {
        return hccbHeliumTransport<Thermo>
        (
            t,
            ht1.viscosityCoefficient_,
            ht1.viscosityTemperatureExponent_,
            ht1.viscosityScale_,
            ht1.conductivityCoefficient_,
            ht1.referenceTemperature_,
            ht1.conductivityTemperatureExponent_,
            ht1.conductivityPressureCoefficient_,
            ht1.conductivityPressureExponent_,
            ht1.conductivityPressureTemperatureExponent_,
            ht1.pressureScale_
        );
    }

    const scalar Y1 = ht1.Y()/t.Y();
    const scalar Y2 = ht2.Y()/t.Y();

    return hccbHeliumTransport<Thermo>
    (
        t,
        Y1*ht1.viscosityCoefficient_ + Y2*ht2.viscosityCoefficient_,
        Y1*ht1.viscosityTemperatureExponent_
      + Y2*ht2.viscosityTemperatureExponent_,
        Y1*ht1.viscosityScale_ + Y2*ht2.viscosityScale_,
        Y1*ht1.conductivityCoefficient_ + Y2*ht2.conductivityCoefficient_,
        Y1*ht1.referenceTemperature_ + Y2*ht2.referenceTemperature_,
        Y1*ht1.conductivityTemperatureExponent_
      + Y2*ht2.conductivityTemperatureExponent_,
        Y1*ht1.conductivityPressureCoefficient_
      + Y2*ht2.conductivityPressureCoefficient_,
        Y1*ht1.conductivityPressureExponent_
      + Y2*ht2.conductivityPressureExponent_,
        Y1*ht1.conductivityPressureTemperatureExponent_
      + Y2*ht2.conductivityPressureTemperatureExponent_,
        Y1*ht1.pressureScale_ + Y2*ht2.pressureScale_
    );
}


template<class Thermo>
inline Foam::hccbHeliumTransport<Thermo> Foam::operator*
(
    const scalar s,
    const hccbHeliumTransport<Thermo>& ht
)
{
    return hccbHeliumTransport<Thermo>
    (
        s*static_cast<const Thermo&>(ht),
        ht.viscosityCoefficient_,
        ht.viscosityTemperatureExponent_,
        ht.viscosityScale_,
        ht.conductivityCoefficient_,
        ht.referenceTemperature_,
        ht.conductivityTemperatureExponent_,
        ht.conductivityPressureCoefficient_,
        ht.conductivityPressureExponent_,
        ht.conductivityPressureTemperatureExponent_,
        ht.pressureScale_
    );
}

// ************************************************************************* //
