
extern crate cgmath;
#[macro_use]
extern crate ndarray;
extern crate image;
extern crate panopaea_util as util;
extern crate generic_array;

use image::GenericImage;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1, Ix2, Ixs};
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
fn down_convolution<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = (output.len() / 2) as isize;

    let filter_size = W::N::to_isize();
    let filter_half = filter_size / 2;
    let low_pass = W::coeff_down_low();
    let high_pass = W::coeff_down_high();

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
fn up_convolution<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = (output.len() / 2) as isize;

    let filter_size = W::N::to_isize();
    let filter_half = filter_size / 2;
    let low_pass = W::coeff_up_low();
    let high_pass = W::coeff_up_high();

    // TODO:
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

fn fwt_1d<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix1>, levels: usize, mut temp: ArrayViewMut<f64, Ix1>) {
    let mut level_size = output.dim();

    // output after decompositions:
    // |-coarse n-|-detail n-|----detail n-1----|-------detail n-2-------| ..
    for n in 0..levels {
        let coarse_slice = s![..level_size as isize];
        let mut src = temp.slice_mut(s![..level_size as isize]);
        let mut dest = output.slice_mut(s![..level_size as isize]);
        src.assign(&dest);

        down_convolution::<W>(src.view(), dest.view_mut());

        level_size = level_size / 2;
    }
}

fn ifwt_1d(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>, levels: usize) {
    // TODO:
}

fn fwt_2d_isotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
    let mut level_size = output.dim();

    for n in 0..levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);

        for i in 0..level_size.0 {
            down_convolution::<W>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }
        src.assign(&dest);
        for i in 0..level_size.1 {
            down_convolution::<W>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }

        level_size = (level_size.0 / 2, level_size.1 / 2);
    }
}

fn fwt_2d_anisotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
    let mut level_size = output.dim();
    for _ in 0..levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);
        for i in 0..level_size.0 {
            down_convolution::<W>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }
        level_size.1 /= 2;
    }

    level_size = output.dim();
    for _ in 0..levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);
        for i in 0..level_size.1 {
            down_convolution::<W>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }
        level_size.0 /= 2;
    }
}

fn main() {
    {
        let mut decomposition = arr1(&[12.0, 4.0, 6.0, 8.0, 4.0, 2.0, 5.0, 7.0]);
        let mut temp = Array1::zeros(decomposition.len());

        fwt_1d::<Haar>(decomposition.view_mut(), 2, temp.view_mut());

        println!("{:?}", decomposition);

        let mut decomposition = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let mut temp = Array1::zeros(decomposition.len());

        fwt_1d::<Haar>(decomposition.view_mut(), 1, temp.view_mut());

        println!("{:?}", decomposition);
    }
    

    {
        let lena = image::open("examples/data/lena.png").unwrap().flipv();
        let mut input = Array2::from_elem((lena.height() as usize, lena.width() as usize), 0.0);
        let mut decomposed = Array2::from_elem((lena.height() as usize, lena.width() as usize), 0.0);
        let mut reconstructed = Array2::from_elem((lena.height() as usize, lena.width() as usize), 0.0);

        // lena to array
        for y in 0..lena.height() {
            for x in 0..lena.width() {
                input[(y as usize, x as usize)] = lena.get_pixel(x, y).data[0] as f64;
                decomposed[(y as usize, x as usize)] = lena.get_pixel(x, y).data[0] as f64;
            }
        }

        fwt_2d_anisotropic::<Bior13>(decomposed.view_mut(), 2, input.view_mut());

        let img_data = {
            let mut data = Vec::new();
            for y in 0 .. decomposed.dim().0 {
                for x in 0 .. decomposed.dim().1 {
                    let val = &decomposed[(y, x)];
                    data.push([
                        util::imgproc::transfer(&val.abs(), 0.0, 1.0*255.0),
                        util::imgproc::transfer(&val.abs(), 0.0, 1.0*255.0),
                        util::imgproc::transfer(&val.abs(), 0.0, 1.0*255.0),
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("lena_fwt.png"),
            &img_data,
            (decomposed.dim().1,
             decomposed.dim().0));

        //one step 2d ifwt
        for i in 0..input.dim().0 {
            up_convolution::<Bior13>(
                decomposed.subview(Axis(0), i as usize),
                reconstructed.subview_mut(Axis(0), i as usize));
        }
        decomposed.assign(&reconstructed);
        for i in 0..input.dim().1 {
            up_convolution::<Bior13>(
                decomposed.subview(Axis(1), i as usize),
                reconstructed.subview_mut(Axis(1), i as usize));
        }

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
            (reconstructed.dim().1,
             reconstructed.dim().0));
    }
}
