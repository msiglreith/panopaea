
//! Smoothing Kernels

use math::Real;
use num::cast;

pub trait Kernel<T: Real> {
    fn w(&self, radius: T) -> T;
    
    /// Gradient factor
    fn grad_w(&self, radius: T) -> T;
    fn laplace_w(&self, radius: T) -> T;
}

/// Poly6 kernel function
///
/// Ref: [MDM03] Sec 3.5
pub struct Poly6<T: Real> {
    h: T,
    w_const: T,
    grad_w_const: T,
}

impl<T: Real> Poly6<T> {
    pub fn new(smoothing_radius: T) -> Self {
        use std::f64;
        let w_frac = cast::<f64, T>(315.0 / 64.0).unwrap();
        let grad_w_frac = cast::<f64, T>(-945.0 / 32.0).unwrap();
        let pi = cast::<f64, T>(f64::consts::PI).unwrap();
        let h9 = smoothing_radius.powi(9);

        Poly6 {
            h: smoothing_radius,
            w_const: w_frac / (pi * h9),
            grad_w_const: grad_w_frac / (pi * h9),
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

    fn grad_w(&self, radius: T) -> T {
        debug_assert!(radius.is_sign_positive());

        if self.h <= radius {
            return T::zero();
        }

        let diff = self.h.powi(2) - radius.powi(2);
        self.grad_w_const * diff.powi(2)
    }

    fn laplace_w(&self, radius: T) -> T {
        unimplemented!()
    }
}

/// Spiky kernel function
///
/// Ref: [MDM03] Sec 3.5
pub struct Spiky<T: Real> {
    h: T,
    w_const: T,
    grad_w_const: T,
}

impl<T: Real> Spiky<T> {
    pub fn new(smoothing_radius: T) -> Self {
        use std::f64;
        let w_frac = cast::<f64, T>(15.0).unwrap();
        let grad_w_frac = cast::<f64, T>(-45.0).unwrap();
        let pi = cast::<f64, T>(f64::consts::PI).unwrap();
        let h6 = smoothing_radius.powi(6);

        Spiky {
            h: smoothing_radius,
            w_const: w_frac / (pi * h6),
            grad_w_const: grad_w_frac / (pi * h6)
        }
    }
}

impl<T: Real> Kernel<T> for Spiky<T> {
    fn w(&self, radius: T) -> T {
        debug_assert!(radius.is_sign_positive());

        if self.h <= radius {
            return T::zero();
        }

        let diff = self.h - radius;
        self.w_const * diff.powi(3)
    }

    fn grad_w(&self, radius: T) -> T {
        debug_assert!(radius.is_sign_positive());

        if self.h <= radius {
            return T::zero();
        }

        let diff = self.h - radius;
        self.grad_w_const * diff.powi(2) / radius
    }

    fn laplace_w(&self, radius: T) -> T {
        unimplemented!()
    }
}
