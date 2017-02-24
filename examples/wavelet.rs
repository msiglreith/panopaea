
extern crate cgmath;
#[macro_use]
extern crate ndarray;
extern crate image;
extern crate panopaea_util as util;
extern crate generic_array;

use image::GenericImage;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1, Ixs};
use generic_array::ArrayLength;
use generic_array::typenum::{U2, U6, Unsigned};

pub trait Wavelet<T> {
    type N: ArrayLength<usize>;
    fn coeff_up_low() -> &'static [T];
    fn coeff_up_high() -> &'static [T];
    fn coeff_down_low() -> &'static [T];
    fn coeff_down_high() -> &'static [T];
}

// Haar wavelet
static HAAR_UP_LOW: &[f64; 2] = &[0.7071067811865476, 0.7071067811865476];
static HAAR_UP_HIGH: &[f64; 2] = &[0.7071067811865476, -0.7071067811865476];
static HAAR_DOWN_LOW: &[f64; 2] = &[0.7071067811865476, 0.7071067811865476];
static HAAR_DOWN_HIGH: &[f64; 2] = &[-0.7071067811865476, 0.7071067811865476];

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

// Biorthogonal 1.3
static BIOR_1_3_UP_LOW: &[f64; 6] = &[0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0];
static BIOR_1_3_UP_HIGH: &[f64; 6] = &[-0.08838834764831845, -0.08838834764831845, 0.7071067811865476, -0.7071067811865476, 0.08838834764831845, 0.08838834764831845];
static BIOR_1_3_DOWN_LOW: &[f64; 6] = &[-0.08838834764831845, 0.08838834764831845, 0.7071067811865476, 0.7071067811865476, 0.08838834764831845, -0.08838834764831845];
static BIOR_1_3_DOWN_HIGH: &[f64; 6] = &[0.0, 0.0, -0.7071067811865476, 0.7071067811865476, 0.0, 0.0];

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

// Array sampler
pub trait Sampler<T: BaseFloat> {
    fn fetch(input: ArrayView<T, Ix1>, idx: Ixs) -> T;
}

pub struct ZeroSampler;
impl Sampler<f64> for ZeroSampler {
    fn fetch(input: ArrayView<f64, Ix1>, idx: Ixs) -> f64 {
        if idx >= 0 && idx < input.len() as isize {
            input[idx as usize]
        } else {
            0.0
        }
    }
}

#[inline]
fn down_convolution(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = (output.len() / 2) as isize;

    let filter_size = <Bior13 as Wavelet<f64>>::N::to_isize();
    let filter_half = filter_size / 2;
    let low_pass = Bior13::coeff_down_low();
    let high_pass = Bior13::coeff_down_high();

    // low-pass
    for i in 0..half_size {
        output[i as usize] = (0..filter_size).fold(0.0, |acc, f| {
            let input_src = 2*i+f-filter_half+1;
            let filter_src = (filter_size-1-f) as usize;
            ZeroSampler::fetch(input, input_src) * low_pass[filter_src] + acc
        });
    }

    // high pass
    for i in 0..half_size {
        output[(i+half_size) as usize] = (0..filter_size).fold(0.0, |acc, f| {
            let input_src = 2*i+f-filter_half+1;
            let filter_src = (filter_size-1-f) as usize;
            ZeroSampler::fetch(input, input_src) * high_pass[filter_src] + acc
        });
    }
}

#[inline]
fn up_convolution(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = (output.len() / 2) as isize;

    let filter_size = <Bior13 as Wavelet<f64>>::N::to_isize();
    let filter_half = filter_size / 2;
    let low_pass = Bior13::coeff_up_low();
    let high_pass = Bior13::coeff_up_high();

    // Index transformation:
    //  for i in 0..half
    //      for f in 0..filter
    //          dest[(2*i+f)%len] += in[i] * low[i] + in[i+half] * high[i];
    //
    //
    output.fill(0.0);
    for i in 0..half_size {
        for f in 0..filter_size {
            let idx_out = (2 * i + f) % output.len() as isize;
            let filter_src = f as usize;
            output[idx_out as usize] +=
                ZeroSampler::fetch(input, i) * low_pass[filter_src] +
                ZeroSampler::fetch(input, i+half_size) * high_pass[filter_src];
        }  
    }
}

fn fwt_1d(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>, levels: usize, mut temp: ArrayViewMut<f64, Ix1>) {
    // early out
    if levels == 0 {
        output.assign(&input);
        return;
    }

    temp.assign(&input);

    // output after decompositions:
    // |-coarse n-|-detail n-|----detail n-1----|-------detail n-2-------| ..

    let mut level_size = input.len();
    for n in 0..levels {
        
        let half_size = level_size / 2;

        down_convolution(temp.view(), output.view_mut());

        // copy coarse data back to temp buffer
        for i in 0..half_size {
            temp[i] = output[i];
        }

        level_size = half_size;
    }
}

fn ifwt_1d(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>, levels: usize) {
    // TODO:
}

fn main() {
    {
        let input = arr1(&[12.0, 4.0, 6.0, 8.0, 4.0, 2.0, 5.0, 7.0]);
        let mut decomposition = Array1::zeros(input.len());
        let mut temp = Array1::zeros(input.len());

        fwt_1d(input.view(), decomposition.view_mut(), 2, temp.view_mut());

        println!("{:?}", decomposition);  
    }
    

    {
        let lena = image::open("examples/data/lena.png").unwrap().flipv();
        let mut output = Array2::from_elem((lena.height() as usize, lena.width() as usize), 0.0);
        let mut reconstructed = Array2::from_elem((lena.height() as usize, lena.width() as usize), 0.0);

        for y in 0..lena.height() {
            let row = (0..lena.width()).into_iter().map(|x| {
                lena.get_pixel(x, y).data[0] as f64
            }).collect::<Vec<_>>();
            let input = arr1(&row);

            down_convolution(input.view(), output.subview_mut(Axis(0), y as usize));
            
        }

        for x in 0..lena.width() {
            let col = (0..lena.height()).into_iter().map(|y| {
                output[(y as usize, x as usize)]
            }).collect::<Vec<_>>();
            let input = arr1(&col);

            down_convolution(input.view(), output.subview_mut(Axis(1), x as usize));
        }

        for x in 0..lena.width() {
            up_convolution(output.subview(Axis(1), x as usize), reconstructed.subview_mut(Axis(1), x as usize));
        }

        for y in 0..lena.height() {
            let row = (0..lena.width()).into_iter().map(|x| {
                reconstructed[(y as usize, x as usize)]
            }).collect::<Vec<_>>();
            let input = arr1(&row);

            up_convolution(input.view(), reconstructed.subview_mut(Axis(0), y as usize));
        }

        let img_data = {
            let mut data = Vec::new();
            for y in 0 .. output.dim().0 {
                for x in 0 .. output.dim().1 {
                    let val = &output[(y, x)];
                    data.push([
                        util::imgproc::transfer(val, -255.0, 255.0),
                        util::imgproc::transfer(val, -255.0, 255.0),
                        util::imgproc::transfer(val, -255.0, 255.0),
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("lena_fwt.png"),
            &img_data,
            output.dim());

        let img_data = {
            let mut data = Vec::new();
            for y in 0 .. reconstructed.dim().0 {
                for x in 0 .. reconstructed.dim().1 {
                    let val = &reconstructed[(y, x)];
                    data.push([
                        util::imgproc::transfer(val, 0.0, 255.0),
                        util::imgproc::transfer(val, 0.0, 255.0),
                        util::imgproc::transfer(val, 0.0, 255.0),
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("lena_ifwt.png"),
            &img_data,
            reconstructed.dim());
    }
}
