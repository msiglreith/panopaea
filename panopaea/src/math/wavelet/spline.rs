
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

static BIOR_1_3_UP_LOW: &'static [f64; 6] = &[0.0, 0.0, 0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903, 0.0, 0.0];
static BIOR_1_3_UP_HIGH: &'static [f64; 6] = &[-0.08838834764831844055010554526310612, -0.08838834764831844055010554526310612, 0.70710678118654752440084436210484903, -0.70710678118654752440084436210484903, 0.08838834764831844055010554526310612, 0.08838834764831844055010554526310612];
static BIOR_1_3_DOWN_LOW: &'static [f64; 6] = &[-0.08838834764831844055010554526310612, 0.08838834764831844055010554526310612, 0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903, 0.08838834764831844055010554526310612, -0.08838834764831844055010554526310612];
static BIOR_1_3_DOWN_HIGH: &'static [f64; 6] = &[0.0, 0.0, -0.70710678118654752440084436210484903, 0.70710678118654752440084436210484903, 0.0, 0.0];


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

static BIOR_2_2_UP_LOW: &'static [f64; 6] = &[0.0, 0.35355339059327376220042218105242451, 0.70710678118654752440084436210484903, 0.35355339059327376220042218105242451, 0.0, 0.0];
static BIOR_2_2_UP_HIGH: &'static [f64; 6] = &[0.0, 0.17677669529663688110021109052621225, 0.35355339059327376220042218105242451, -1.06066017177982128660126654315727355, 0.35355339059327376220042218105242451, 0.17677669529663688110021109052621225];
static BIOR_2_2_DOWN_LOW: &'static [f64; 6] = &[0.0, -0.17677669529663688110021109052621225, 0.35355339059327376220042218105242451, 1.06066017177982128660126654315727355, 0.35355339059327376220042218105242451, -0.17677669529663688110021109052621225];
static BIOR_2_2_DOWN_HIGH: &'static [f64; 6] = &[0.0, 0.35355339059327376220042218105242451, -0.70710678118654752440084436210484903, 0.35355339059327376220042218105242451, 0.0, 0.0];


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

static BIOR_2_4_UP_LOW: &'static [f64; 10] = &[
    0.0, 0.0, 0.0, 0.35355339059327376220042218105242451, 0.70710678118654752440084436210484903,
    0.35355339059327376220042218105242451, 0.0, 0.0, 0.0, 0.0];

static BIOR_2_4_UP_HIGH: &'static [f64; 10] = &[
    0.0, -0.03314563036811942, -0.06629126073623884, 0.17677669529663688110021109052621225, 0.4198446513295126,
    -0.9943689110435825, 0.4198446513295126, 0.17677669529663688110021109052621225, -0.06629126073623884, -0.03314563036811942];

static BIOR_2_4_DOWN_LOW: &'static [f64; 10] = &[
    0.0, 0.03314563036811942, -0.06629126073623884, -0.17677669529663688110021109052621225, 0.4198446513295126,
    0.9943689110435825, 0.4198446513295126, -0.17677669529663688110021109052621225, -0.06629126073623884, 0.03314563036811942];

static BIOR_2_4_DOWN_HIGH: &'static [f64; 10] = &[
    0.0, 0.0, 0.0, 0.35355339059327376220042218105242451, -0.70710678118654752440084436210484903,
    0.35355339059327376220042218105242451, 0.0, 0.0, 0.0, 0.0
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

static BIOR_3_1_UP_LOW: &'static [f64; 4] = &[0.17677669529663688110021109052621225, 0.5303300858899107, 0.5303300858899107, 0.17677669529663688110021109052621225];
static BIOR_3_1_UP_HIGH: &'static [f64; 4] = &[-0.35355339059327376220042218105242451, -1.06066017177982128660126654315727355, 1.06066017177982128660126654315727355, 0.35355339059327376220042218105242451];
static BIOR_3_1_DOWN_LOW: &'static [f64; 4] = &[-0.35355339059327376220042218105242451, 1.06066017177982128660126654315727355, 1.06066017177982128660126654315727355, -0.35355339059327376220042218105242451];
static BIOR_3_1_DOWN_HIGH: &'static [f64; 4] = &[-0.17677669529663688110021109052621225, 0.5303300858899107, -0.5303300858899107, 0.17677669529663688110021109052621225];