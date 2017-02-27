
use generic_array::typenum::{U4, U6, U10};
use super::Wavelet;

/// Biorthogonal spline wavelet (1.3)
pub struct Bior13;
impl Wavelet<f64> for Bior13 {
    type N = U6;
    #[inline]
    fn coeff_up_low() -> &'static [f64] { BIOR_1_3_UP_LOW }
    #[inline]
    fn coeff_up_high() -> &'static [f64] { BIOR_1_3_UP_HIGH }
    #[inline]
    fn coeff_down_low() -> &'static [f64] { BIOR_1_3_DOWN_LOW }
    #[inline]
    fn coeff_down_high() -> &'static [f64] { BIOR_1_3_DOWN_HIGH }
}

static BIOR_1_3_UP_LOW: &[f64; 6] = &[0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0];
static BIOR_1_3_UP_HIGH: &[f64; 6] = &[-0.08838834764831845, -0.08838834764831845, 0.7071067811865476, -0.7071067811865476, 0.08838834764831845, 0.08838834764831845];
static BIOR_1_3_DOWN_LOW: &[f64; 6] = &[-0.08838834764831845, 0.08838834764831845, 0.7071067811865476, 0.7071067811865476, 0.08838834764831845, -0.08838834764831845];
static BIOR_1_3_DOWN_HIGH: &[f64; 6] = &[0.0, 0.0, -0.7071067811865476, 0.7071067811865476, 0.0, 0.0];


/// Biorthogonal spline wavelet (2.2)
pub struct Bior22;
impl Wavelet<f64> for Bior22 {
    type N = U6;
    #[inline]
    fn coeff_up_low() -> &'static [f64] { BIOR_2_2_UP_LOW }
    #[inline]
    fn coeff_up_high() -> &'static [f64] { BIOR_2_2_UP_HIGH }
    #[inline]
    fn coeff_down_low() -> &'static [f64] { BIOR_2_2_DOWN_LOW }
    #[inline]
    fn coeff_down_high() -> &'static [f64] { BIOR_2_2_DOWN_HIGH }
}

static BIOR_2_2_UP_LOW: &[f64; 6] = &[0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738, 0.0, 0.0];
static BIOR_2_2_UP_HIGH: &[f64; 6] = &[0.0, 0.1767766952966369, 0.3535533905932738, -1.0606601717798214, 0.3535533905932738, 0.1767766952966369];
static BIOR_2_2_DOWN_LOW: &[f64; 6] = &[0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798214, 0.3535533905932738, -0.1767766952966369];
static BIOR_2_2_DOWN_HIGH: &[f64; 6] = &[0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, 0.0, 0.0];


/// Biorthogonal spline wavelet (2.4)
pub struct Bior24;
impl Wavelet<f64> for Bior24 {
    type N = U10;
    #[inline]
    fn coeff_up_low() -> &'static [f64] { BIOR_2_4_UP_LOW }
    #[inline]
    fn coeff_up_high() -> &'static [f64] { BIOR_2_4_UP_HIGH }
    #[inline]
    fn coeff_down_low() -> &'static [f64] { BIOR_2_4_DOWN_LOW }
    #[inline]
    fn coeff_down_high() -> &'static [f64] { BIOR_2_4_DOWN_HIGH }
}

static BIOR_2_4_UP_LOW: &[f64; 10] = &[
    0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865476,
    0.3535533905932738, 0.0, 0.0, 0.0, 0.0];

static BIOR_2_4_UP_HIGH: &[f64; 10] = &[
    0.0, -0.03314563036811942, -0.06629126073623884, 0.1767766952966369, 0.4198446513295126,
    -0.9943689110435825, 0.4198446513295126, 0.1767766952966369, -0.06629126073623884, -0.03314563036811942];

static BIOR_2_4_DOWN_LOW: &[f64; 10] = &[
    0.0, 0.03314563036811942, -0.06629126073623884, -0.1767766952966369, 0.4198446513295126,
    0.9943689110435825, 0.4198446513295126, -0.1767766952966369, -0.06629126073623884, 0.03314563036811942];

static BIOR_2_4_DOWN_HIGH: &[f64; 10] = &[
    0.0, 0.0, 0.0, 0.3535533905932738, -0.7071067811865476,
    0.3535533905932738, 0.0, 0.0, 0.0, 0.0
];


/// Biorthogonal spline wavelet (3.1)
pub struct Bior31;
impl Wavelet<f64> for Bior31 {
    type N = U4;
    #[inline]
    fn coeff_up_low() -> &'static [f64] { BIOR_3_1_UP_LOW }
    #[inline]
    fn coeff_up_high() -> &'static [f64] { BIOR_3_1_UP_HIGH }
    #[inline]
    fn coeff_down_low() -> &'static [f64] { BIOR_3_1_DOWN_LOW }
    #[inline]
    fn coeff_down_high() -> &'static [f64] { BIOR_3_1_DOWN_HIGH }
}

static BIOR_3_1_UP_LOW: &[f64; 4] = &[0.1767766952966369, 0.5303300858899107, 0.5303300858899107, 0.1767766952966369];
static BIOR_3_1_UP_HIGH: &[f64; 4] = &[-0.3535533905932738, -1.0606601717798214, 1.0606601717798214, 0.3535533905932738];
static BIOR_3_1_DOWN_LOW: &[f64; 4] = &[-0.3535533905932738, 1.0606601717798214, 1.0606601717798214, -0.3535533905932738];
static BIOR_3_1_DOWN_HIGH: &[f64; 4] = &[-0.1767766952966369, 0.5303300858899107, 0.5303300858899107, 0.1767766952966369];
