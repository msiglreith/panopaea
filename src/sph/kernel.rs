
//! Smoothing Kernels

use math::Real;
use num::FromPrimitive;
use num::cast;

pub trait Kernel<T: Real> {
    fn w(&self, radius: T) -> T;
}

/// Poly6 kernel function
/// Source: Mueller, M., Charypar, D., Gross, M., Particle-Based Fluid Simulation for Interactive Applications, 2003, SIGGRAPH
pub struct Poly6<T: Real> {
    h: T,
    w_const: T,
}

impl<T: Real> Poly6<T> {
    pub fn new(smoothing_radius: T) -> Self {
        use std::f64;
        let frac = cast::<f64, T>(315.0 / 64.0).unwrap();
        let pi = cast::<f64, T>(f64::consts::PI).unwrap();
        let h9 = smoothing_radius.powi(9);

        Poly6 {
            h: smoothing_radius,
            w_const: frac / (pi * h9),
        }
    }
}

impl<T: Real> Kernel<T> for Poly6<T> {
    fn w(&self, radius: T) -> T {
        debug_assert!(radius.is_sign_positive());

        if self.h <= radius {
            return T::zero();
        }

        let diff = self.h.powi(2) - radius.powi(2);
        self.w_const * diff.powi(3)
    }
}
