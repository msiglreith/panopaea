
use generic_array::ArrayLength;
use generic_array::typenum::Unsigned;

pub mod haar;
pub mod spline;

pub trait Wavelet<T> {
    type N: ArrayLength<usize>;
    fn filter_length() -> isize { Self::N::to_isize() }
    fn border_width() -> isize { Self::N::to_isize() }
    fn coeff_up_low() -> &'static [T];
    fn coeff_up_high() -> &'static [T];
    fn coeff_down_low() -> &'static [T];
    fn coeff_down_high() -> &'static [T];
}
