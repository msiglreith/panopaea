
use cgmath::{self, InnerSpace};
use math::Real;
use ndarray::{Array2};
use num::cast;
use rand;
use rand::distributions::normal;

use std::f64::consts::PI;

/// Representing a spectral density function of angular frequency.
///
/// This is only the non-directional component of the spectrum following [Horvath15].
pub trait Spectrum<T: Real>: Sync {
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

pub struct Parameters<T> {
    pub surface_tension: T,
    pub water_density: T,
    pub water_depth: T,
    pub gravity: T, // [m/s^2]
}

pub fn build_height_spectrum<S, T>(
    parameters: &Parameters<T>,
    spectrum: &S,
    domain_size: T,
    resolution: usize) -> Array2<(T, T)>
where
    S: Spectrum<T>,
    T: Real
{
    let pi: T = cast(PI).unwrap();
    let mut height_spectrum = Array2::from_elem((resolution+1, resolution+1), (T::zero(), T::zero()));
    par_azip!(index (i, j), mut height_spectrum in {
        let x: T = cast(2 * i as isize - resolution as isize).unwrap();
        let y: T = cast(2 * j as isize - resolution as isize).unwrap();

        if i == resolution/2 && j == resolution/2 {
            *height_spectrum = (T::zero(), T::zero());
        } else {
            let k = cgmath::vec2(
                pi * x / domain_size,
                pi * y / domain_size,
            );
            let (sample, _omega) = sample_spectrum(parameters, spectrum, k, domain_size);
            *height_spectrum = sample;
        };
    });

    height_spectrum
}

fn sample_spectrum<S, T>(
    parameters: &Parameters<T>,
    spectrum: &S,
    pos: cgmath::Vector2<T>,
    domain_size: T) -> ((T, T), T)
where
    S: Spectrum<T>,
    T: Real
{
    assert!(pos.magnitude() > T::default_epsilon());

    let two: T = cast(2.0f64).unwrap();
    let pi: T = cast(PI).unwrap();

    let theta = (pos.y).atan2(pos.x);
    let grad_k = two * pi / domain_size;

    let (omega, grad_omega) = dispersion_capillary(parameters, pos.magnitude());
    let spreading = directional_spreading(parameters, omega, theta, directional_base_donelan_banner);
    let sample = spectrum.evaluate(omega);

    let normal::StandardNormal(z) = rand::random();
    let phase = two * pi * rand::random::<T>();

    let amplitude = cast::<_, T>(z).unwrap() * (two * spreading * sample * grad_k.powi(2) * grad_omega / pos.magnitude()).sqrt();

    ((phase.cos() * amplitude, phase.sin() * amplitude), omega)
}


fn dispersion_capillary<T>(parameters: &Parameters<T>, wave_number: T) -> (T, T)
where
    T: Real
{
    let sech = |x: T| { T::one() / x.cosh() };
    let two: T = cast(2.0f64).unwrap();
    let three: T = cast(3.0f64).unwrap();

    let sigma = parameters.surface_tension;
    let rho = parameters.water_density;
    let g = parameters.gravity;
    let h = parameters.water_depth;
    let k = wave_number;

    let dispersion = ((g*k + (sigma/rho) * k.powi(3)) * (h*k).tanh()).sqrt();
    let grad_dispersion = (
            h * sech(h*k).powi(2) * (g*k + (sigma/rho) * k.powi(3)) +
            (h*k).tanh() * (g + three*(sigma/rho) * k.powi(2))
        ) / (two * dispersion);

    (dispersion, grad_dispersion)
}

fn directional_spreading<F, T>(parameters: &Parameters<T>, omega: T, theta: T, directional_base: F) -> T
where
    F: Fn(&Parameters<T>, T, T) -> T,
    T: Real,
{
    unimplemented!()
}

fn directional_base_donelan_banner<T>(params: &Parameters<T>, omega: T, theta: T) -> T
where
    T: Real,
{
    unimplemented!()
}
