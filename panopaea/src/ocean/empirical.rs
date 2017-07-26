
use math::Real;
use num::cast;

/// Representing a spectral density function of angular frequency.
///
/// This is only the non-directional component of the spectrum following [Horvath15].
pub trait Spectrum<T: Real> {
    fn evaluate(&self, omega: T) -> T;
}


/// Joint North Sea Wave Observation Project (JONSWAP) Spectrum [Horvath15] Section 5.1.4
pub struct SpectrumJONSWAP<T: Real> {
    pub wind_speed: T, // [m/s]
    pub fetch: T,
    pub gravity: T,    // [m/s^2]
}

impl<T: Real> SpectrumJONSWAP<T> {
    fn dispersion_peak(&self) -> T {
        let third: T = cast(1.0/3.0f64).unwrap();
        let factor: T = cast(22.0f64).unwrap();

        // NOTE: pow(x, 1/3) missing in [Horvath2015]
        factor * (self.gravity.powi(2) / (self.wind_speed * self.fetch)).powf(third)
    }
}

impl<T: Real> Spectrum<T> for SpectrumJONSWAP<T> {
    // [Horvath15] Eq. 28
    fn evaluate(&self, omega: T) -> T {
        if omega < T::default_epsilon() {
            return T::zero();
        }

        let two: T = cast(2.0f64).unwrap();
        let gamma: T = cast(3.3f64).unwrap();
        let omega_peak = self.dispersion_peak();
        let alpha = {
            let factor: T = cast(0.076f64).unwrap();
            let exponent: T = cast(0.22f64).unwrap();

            factor * (self.wind_speed.powi(2) / (self.fetch * self.gravity)).powf(exponent)
        };
        let sigma: T = {
            let sigma = if omega <= omega_peak { 0.07 } else { 0.09 };
            cast(sigma).unwrap()
        };
        let r = (-(omega - omega_peak).powi(2) / (two * (sigma * omega_peak).powi(2))).exp();
        let factor: T = cast(-5.0/4.0).unwrap();

        (alpha * (self.gravity).powi(2) / omega.powi(5)) * (factor * (omega_peak/omega).powi(4)).exp() * gamma.powf(r)
    }
}

/// Texel MARSEN ARSLOE (TMA) Spectrum [Horvath15] Section 5.1.5
pub struct SpectrumTMA<T: Real> {
    pub jonswap: SpectrumJONSWAP<T>,
    pub depth: T, // TODO: [m]
}

impl<T: Real> SpectrumTMA<T> {
    /// Kitaigorodskii Depth Attenuation Function [Horvath15] Eq. 29
    ///
    /// Using the approximation from Thompson and Vincent, 1983
    /// as proposed in Section 5.1.5.
    fn kitaigorodskii_depth_attenuation(&self, omega: T) -> T {
        let half: T = cast(0.5f64).unwrap();
        let two: T = cast(2.0f64).unwrap();

        let omega_h = (omega * (self.depth / self.jonswap.gravity)).max(T::zero()).min(two);
        if omega_h <= T::one() {
            half * omega_h.powi(2)
        } else {
            T::one() - half * (two - omega_h).powi(2)
        }
    }
}

impl<T: Real> Spectrum<T> for SpectrumTMA<T> {
    fn evaluate(&self, omega: T) -> T {
        self.jonswap.evaluate(omega) * self.kitaigorodskii_depth_attenuation(omega)
    }
}
