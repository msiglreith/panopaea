
extern crate cgmath;
#[macro_use]
extern crate ndarray;
extern crate image;
extern crate panopaea;
extern crate panopaea_util as util;
extern crate generic_array;

use image::GenericImage;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1, Ix2, Ix, Ixs};
use generic_array::typenum::{Unsigned};

use panopaea::wavelet::{haar, spline, Wavelet};

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

    // TODO: FIXME:
    // Index transformation:
    //  for i in 0..half
    //      for f in 0..filter
    //          dest[(2*i+f)%len] += in[i] * low[i] + in[i+half] * high[i];
    //
    //
    output.fill(0.0);
    for i in 0..half_size {
        for f in 0..filter_size {
            let idx_out = {
                let idx = (2 * i + f - filter_half + 1);
                if idx < 0 {
                    (idx + output.len() as isize) % output.len() as isize
                } else {
                    idx % output.len() as isize
                }
            };
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
    for _ in 0..levels {
        let coarse_slice = s![..level_size as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);

        down_convolution::<W>(src.view(), dest.view_mut());

        level_size /= 2;
    }
}

fn ifwt_1d<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix1>, levels: usize, mut temp: ArrayViewMut<f64, Ix1>) {
    let mut level_size = output.dim() / 2usize.pow(levels as u32);

    for _ in 0..levels {
        level_size *= 2;
        let slice = s![..level_size as isize];
        let mut src = temp.slice_mut(slice);
        let mut dest = output.slice_mut(slice);
        src.assign(&dest);

        up_convolution::<W>(src.view(), dest.view_mut());
    }
}

fn fwt_2d_isotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
    let mut level_size = output.dim();

    for n in 0..levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);

        // x direction
        for i in 0..level_size.0 {
            down_convolution::<W>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }
        src.assign(&dest);

        // y direction
        for i in 0..level_size.1 {
            down_convolution::<W>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }

        level_size = (level_size.0 / 2, level_size.1 / 2);
    }
}

fn ifwt_2d_isotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
    let level_pow = 2usize.pow(levels as u32);
    let mut level_size = output.dim();
    level_size = (level_size.0 / level_pow, level_size.1 / level_pow);

    for _ in 0..levels {
        level_size = (level_size.0 * 2, level_size.1 * 2);
        let slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(slice);
        let mut dest = output.slice_mut(slice);
        src.assign(&dest);

        println!("{:?}", (src.dim(), dest.dim()));

        // y direction
        for i in 0..level_size.1 {
            up_convolution::<W>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }

        src.assign(&dest);

        // x direction
        for i in 0..level_size.0 {
            up_convolution::<W>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }
    }
}

fn fwt_2d_separate_isotropic<Wx, Wy>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>)
    where Wx: Wavelet<f64>, Wy: Wavelet<f64> 
{
    let mut level_size = output.dim();

    for n in 0..levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(coarse_slice);
        let mut dest = output.slice_mut(coarse_slice);
        src.assign(&dest);

        // x direction
        for i in 0..level_size.0 {
            down_convolution::<Wx>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }
        src.assign(&dest);

        // y direction
        for i in 0..level_size.1 {
            down_convolution::<Wy>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }

        level_size = (level_size.0 / 2, level_size.1 / 2);
    }
}

fn ifwt_2d_separate_isotropic<Wx, Wy>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>)
    where Wx: Wavelet<f64>, Wy: Wavelet<f64>
{
    let level_pow = 2usize.pow(levels as u32);
    let mut level_size = output.dim();
    level_size = (level_size.0 / level_pow, level_size.1 / level_pow);

    for _ in 0..levels {
        level_size = (level_size.0 * 2, level_size.1 * 2);
        let slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let mut src = temp.slice_mut(slice);
        let mut dest = output.slice_mut(slice);
        src.assign(&dest);

        println!("{:?}", (src.dim(), dest.dim()));

        // y direction
        for i in 0..level_size.1 {
            up_convolution::<Wx>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }

        src.assign(&dest);

        // x direction
        for i in 0..level_size.0 {
            up_convolution::<Wy>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }
    }
}

fn fwt_2d_anisotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
    // x direction
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

    // y direction
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

fn div_coeff_2d(levels: usize,
    src_vx: ArrayView<f64, Ix2>, mut dst_vx: ArrayViewMut<f64, Ix2>,
    src_vy: ArrayView<f64, Ix2>, mut dst_vy: ArrayViewMut<f64, Ix2>)
{
    debug_assert!(src_vx.dim() == dest_vx.dim(),
        "Input arrays differ in dimension (vx: {:?}, vy: {:?})",
        src_vx.dim(), src_vx.dim());
    debug_assert!(src_vx.dim() == dest_vx.dim(),
        "src and dest (vx) differ in dimension (src: {:?}, dest: {:?})",
        src_vx.dim(), dest_vx.dim());
    debug_assert!(src_vy.dim() == dest_vy.dim(),
        "src and dest (vy) differ in dimension (src: {:?}, dest: {:?})",
        src_vy.dim(), dest_vy.dim());


    // size of a filtered slice
    let mut level_size = src_vx.dim();

    for _ in 0 .. levels {
        let mut level_size = (level_size.0 / 2, level_size.1 / 2);

        // split into wavelet components
        let (mut s_vx_00, mut s_vx_01, mut s_vx_10, mut s_vx_11) = {
            let (mut low_y, mut high_y) = src_dx.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut s_vx_00, mut s_vx_01, mut s_vx_10, mut s_vx_11) = {
            let (mut low_y, mut high_y) = src_vx.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut d_vx_00, mut d_vx_01, mut d_vx_10, mut d_vx_11) = {
            let (mut low_y, mut high_y) = dst_vx.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut d_vy_00, mut d_vy_01, mut d_vy_10, mut d_vy_11) = {
            let (mut low_y, mut high_y) = dst_vy.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        // TODO: generate new div wavelet coefficients
        
    }
}

fn main() {
    {
        let mut decomposition = arr1(&[12.0, 4.0, 6.0, 8.0, 4.0, 2.0, 5.0, 7.0]);
        let mut temp = Array1::zeros(decomposition.len());

        fwt_1d::<spline::Bior22>(decomposition.view_mut(), 2, temp.view_mut());

        println!("{:?}", decomposition);

        ifwt_1d::<spline::Bior22>(decomposition.view_mut(), 2, temp.view_mut());

        println!("{:?}", decomposition);

        let mut decomposition = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let mut temp = Array1::zeros(decomposition.len());

        fwt_1d::<haar::Haar>(decomposition.view_mut(), 1, temp.view_mut());

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

        fwt_2d_separate_isotropic::<spline::Bior31, spline::Bior31>(decomposed.view_mut(), 1, input.view_mut());

        let img_data = {
            let mut data = Vec::new();
            for y in 0 .. decomposed.dim().0 {
                for x in 0 .. decomposed.dim().1 {
                    let val = &decomposed[(y, x)];
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
            format!("lena_fwt.png"),
            &img_data,
            (decomposed.dim().1,
             decomposed.dim().0));

        ifwt_2d_separate_isotropic::<spline::Bior31, spline::Bior31>(decomposed.view_mut(), 1, input.view_mut());

        let img_data = {
            let mut data = Vec::new();
            for y in 0 .. decomposed.dim().0 {
                for x in 0 .. decomposed.dim().1 {
                    let val = &decomposed[(y, x)];
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
            (decomposed.dim().1,
             decomposed.dim().0));
    }
}
