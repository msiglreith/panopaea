
use generic_array::ArrayLength;

pub mod haar;
pub mod spline;

pub trait Wavelet<T> {
    type N: ArrayLength<usize>;
    fn coeff_up_low() -> &'static [T];
    fn coeff_up_high() -> &'static [T];
    fn coeff_down_low() -> &'static [T];
    fn coeff_down_high() -> &'static [T];
}
