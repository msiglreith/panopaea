
//! Smoothing Kernels

use alga::general::Real;

/// Poly6 kernel function
///
/// Source: Mueller, M., Charypar, D., Gross, M., Particle-Based Fluid Simulation for Interactive Applications, 2003, SIGGRAPH
pub fn poly6<T: Real>(radius: T, h: T) -> T {
    debug_assert!(radius.is_sign_positive());

    if h <= radius {
        return T::zero();
    }

    let frac = T::from_subset(&(315.0 / 64.0));
    let pi = T::pi();
    let h9 = h.powi(9);
    let diff = h.powi(2) - radius.powi(2);

    frac * (pi * h9) * diff.powi(3)
}
