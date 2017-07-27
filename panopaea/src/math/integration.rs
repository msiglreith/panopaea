
use super::Real;

pub fn trapezoidal_quadrature<F, T>(interval: (T, T), substeps: usize, func: F) -> T
where
    F: Fn(T) -> T,
    T: Real,
{
    let interval_length = interval.1 - interval.0;
    let substeps = 128;
    let substep_length = interval_length / T::new(substeps);

    let integral =
        (0..substeps+1).fold(T::zero(), |integral, i| {
            integral + func(interval.0 + (T::new(i) * substep_length))
        });

    substep_length * integral
}
