
use cgmath::{self, InnerSpace};
use math::Real;
use ndarray::{Array2};
use rand;
use rand::distributions::normal;

use std::f32::consts::PI;

fn dispersion_peak<T: Real>(gravity: T, wind_speed: T, fetch: T) -> T {
    // Note: pow(x, 1/3) is missing in [Horvath2015]
    T::new(22.0) * (gravity.powi(2) / (wind_speed * fetch)).powf(T::new(1.0/3.0))
}

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

impl<T: Real> Spectrum<T> for SpectrumJONSWAP<T> {
    // [Horvath15] Eq. 28
    fn evaluate(&self, omega: T) -> T {
        if omega < T::default_epsilon() {
            return T::zero();
        }

        let gamma = T::new(3.3);
        let omega_peak = dispersion_peak(self.gravity, self.wind_speed, self.fetch);
        let alpha =
            T::new(0.076) * (self.wind_speed.powi(2) / (self.fetch * self.gravity)).powf(T::new(0.22));
        let sigma = T::new(if omega <= omega_peak { 0.07 } else { 0.09 });
        let r = (-(omega - omega_peak).powi(2) / (T::new(2.0) * (sigma * omega_peak).powi(2))).exp();

        (alpha * (self.gravity).powi(2) / omega.powi(5)) * (T::new(-5.0/4.0) * (omega_peak/omega).powi(4)).exp() * gamma.powf(r)
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
        let omega_h = (omega * (self.depth / self.jonswap.gravity)).max(T::zero()).min(T::new(2.0));
        if omega_h <= T::one() {
            T::new(0.5) * omega_h.powi(2)
        } else {
            T::one() - T::new(0.5) * (T::new(2.0) - omega_h).powi(2)
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
    pub wind_speed: T, // [m/s]
    pub fetch: T,
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
    let pi = T::new(PI);
    let mut height_spectrum = Array2::from_elem((resolution+1, resolution+1), (T::zero(), T::zero()));
    par_azip!(index (i, j), mut height_spectrum in {
        let x: T = T::new(2 * i as isize - resolution as isize);
        let y: T = T::new(2 * j as isize - resolution as isize);

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

    let theta = (pos.y).atan2(pos.x);
    let grad_k = T::new(2.0 * PI) / domain_size;

    let (omega, grad_omega) = dispersion_capillary(parameters, pos.magnitude());
    let spreading = directional_spreading(parameters, omega, theta, directional_base_donelan_banner);
    let sample = spectrum.evaluate(omega);

    let normal::StandardNormal(z) = rand::random();
    let phase = T::new(2.0 * PI) * rand::random::<T>();

    let amplitude = T::new(z as f32) * (T::new(2.0) * spreading * sample * grad_k.powi(2) * grad_omega / pos.magnitude()).sqrt();

    ((phase.cos() * amplitude, phase.sin() * amplitude), omega)
}


fn dispersion_capillary<T>(parameters: &Parameters<T>, wave_number: T) -> (T, T)
where
    T: Real
{
    let sech = |x: T| { T::one() / x.cosh() };

    let sigma = parameters.surface_tension;
    let rho = parameters.water_density;
    let g = parameters.gravity;
    let h = parameters.water_depth;
    let k = wave_number;

    let dispersion = ((g*k + (sigma/rho) * k.powi(3)) * (h*k).tanh()).sqrt();
    let grad_dispersion = (
            h * sech(h*k).powi(2) * (g*k + (sigma/rho) * k.powi(3)) +
            (h*k).tanh() * (g + T::new(3.0)*(sigma/rho) * k.powi(2))
        ) / (T::new(2.0) * dispersion);

    (dispersion, grad_dispersion)
}

fn directional_spreading<F, T>(parameters: &Parameters<T>, omega: T, theta: T, directional_base: F) -> T
where
    F: Fn(&Parameters<T>, T, T) -> T,
    T: Real,
{
    unimplemented!()
}

// Donelan-Banner Directional Spreading [Horvath15] Eq. 38
fn directional_base_donelan_banner<T>(parameters: &Parameters<T>, omega: T, theta: T) -> T
where
    T: Real,
{
    let beta = {
        let omega_peak = dispersion_peak(parameters.gravity, parameters.wind_speed, parameters.fetch);
        let omega_ratio = omega/omega_peak;

        if omega_ratio < T::new(0.95) {
            T::new(2.61) * omega_ratio.powf(T::new(1.3))
        } else if omega_ratio < T::new(1.6) {
            T::new(2.28) * omega_ratio.powf(T::new(-1.3))
        } else {
            let epsilon = T::new(-0.4) + T::new(0.8393) * (T::new(-0.567) * (omega_ratio.powi(2)).ln()).exp();
            T::new(10).powf(epsilon)
        }
    };

    let sech = |x: T| { T::one() / x.cosh() };

    beta / (T::new(2.0) * (beta * T::new(PI)).tanh()) * sech(beta * theta).powi(2)
}
