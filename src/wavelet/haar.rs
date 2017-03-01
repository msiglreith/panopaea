
use generic_array::typenum::U2;

use super::Wavelet;


// Haar wavelet
static HAAR_UP_LOW: &[f64; 2] = &[0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903];
static HAAR_UP_HIGH: &[f64; 2] = &[0.70710678118654752440084436210484903, -0.70710678118654752440084436210484903];
static HAAR_DOWN_LOW: &[f64; 2] = &[0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903];
static HAAR_DOWN_HIGH: &[f64; 2] = &[-0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903];

pub struct Haar;
impl Wavelet<f64> for Haar {
    type N = U2;
    #[inline]
    fn coeff_up_low() -> &'static [f64] { HAAR_UP_LOW }
    #[inline]
    fn coeff_up_high() -> &'static [f64] { HAAR_UP_HIGH }
    #[inline]
    fn coeff_down_low() -> &'static [f64] { HAAR_DOWN_LOW }
    #[inline]
    fn coeff_down_high() -> &'static [f64] { HAAR_DOWN_HIGH }
}
