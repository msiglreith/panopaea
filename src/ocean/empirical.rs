
use math::Real;

/// Representing a spectral density function of angular frequency.
///
/// This is only the non-directional component of the spectrum following [Horvath15].
pub trait Spectrum<T: Real> {
    fn evaluate(&self, omega: Real) -> Real;
}

/// Kitaigorodskii Depth Attenuation Function [Horvath15] Eq. 29
///
/// Using the approximation from Thompson and Vincent, 1983
/// as proposed in Section 5.1.5.
fn kitaigorodskii_depth_attenuation<T: Real>(omega: T, depth: T, gravity: T) -> T {
    let omega_h = (omega * (depth / gravity)).max(T::zero()).min(cast::<f64, T>(2.0).unwrap());

    // TODO
    unimplemented!()
}

/// Texel MARSEN ARSLOE (TMA) Spectrum [Horvath15] Section 5.1.5
pub struct SpectrumTMA;

impl<T: Real> Spectrum<T> for SpectrumTMA {
    fn evaluate(&self, omega: Real) -> Real {
        unimplemented!()
    }
}
