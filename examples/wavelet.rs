
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
    src_vx: ArrayView<f64, Ix2>, src_vy: ArrayView<f64, Ix2>,
    mut dst_div: ArrayViewMut<f64, Ix2>, mut dst_n: ArrayViewMut<f64, Ix2>)
{
    // all arrays should have the same dimension
    debug_assert!(src_vx.dim() == src_vy.dim(),
        "Input arrays differ in dimension (vx: {:?}, vy: {:?})",
        src_vx.dim(), src_vx.dim());
    debug_assert!(src_vx.dim() == dst_div.dim(),
        "src (vx) and dst (div) differ in dimension (src: {:?}, dst: {:?})",
        src_vx.dim(), dst_div.dim());
    debug_assert!(src_vy.dim() == dst_n.dim(),
        "src (vy) and dst (n) differ in dimension (src: {:?}, dst: {:?})",
        src_vy.dim(), dst_n.dim());


    // size of a filtered slice
    let mut level_size = src_vx.dim();

    for n in 0 .. levels {
        let coarse_slice = s![..level_size.0 as isize, ..level_size.1 as isize];
        let src_vx = src_vx.slice(coarse_slice);
        let src_vy = src_vy.slice(coarse_slice);
        let mut dst_div = dst_div.slice_mut(coarse_slice);
        let mut dst_n = dst_n.slice_mut(coarse_slice);

        level_size = (level_size.0 / 2, level_size.1 / 2);

        // split into wavelet components
        let (mut vx_00, mut vx_01, mut vx_10, mut vx_11) = {
            let (mut low_y, mut high_y) = src_vx.view().split_at(Axis(0), level_size.0);
            println!("{:?}", (low_y.dim(), high_y.dim()));
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        println!("{:?}", (vx_00.dim(), vx_01.dim(), vx_10.dim(), vx_11.dim()));

        let (mut vy_00, mut vy_01, mut vy_10, mut vy_11) = {
            let (mut low_y, mut high_y) = src_vy.view().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut div_00, mut div_01, mut div_10, mut div_11) = {
            let (mut low_y, mut high_y) = dst_div.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut n_00, mut n_01, mut n_10, mut n_11) = {
            let (mut low_y, mut high_y) = dst_n.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        // TODO: generate new div wavelet coefficients
        for y in 0..level_size.0 {
            for x in 0..level_size.1 {
                div_01[(y, x)] = vy_01[(y, x)];
                div_10[(y, x)] = vx_10[(y, x)];
                div_11[(y, x)] = (vx_11[(y, x)] - vy_11[(y, x)]) / 2.0;

                n_01[(y, x)] = vx_01[(y, x)] + (vy_01[(y, x)] - vy_01[(y.saturating_sub(1), x)]) / 4.0; // TODO
                n_10[(y, x)] = vy_10[(y, x)] + (vx_10[(y, x)] - vx_10[(y, x.saturating_sub(1))]) / 4.0; // TODO
                n_11[(y, x)] = (vx_11[(y, x)] + vy_11[(y, x)]) / 2.0;
            }
        }

        if n == levels-1 {
            div_00.assign(&vx_00);
            n_00.assign(&vy_00);
        }
    }
}

fn main() {
    if false {
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
    

    // example:
    // fwt/ifwt of a 2d image
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

    // example:
    // extracting divergence coefficients from velocity field
    {
        let vel_input = image::open("vel_sample_300.png").unwrap().flipv();
        let mut vx = Array2::from_elem((vel_input.height() as usize, vel_input.width() as usize), 0.0);
        let mut vy = Array2::from_elem((vel_input.height() as usize, vel_input.width() as usize), 0.0);
        let mut temp = Array2::from_elem((vel_input.height() as usize, vel_input.width() as usize), 0.0);

        let mut div = Array2::from_elem((vel_input.height() as usize, vel_input.width() as usize), 0.0);
        let mut n = Array2::from_elem((vel_input.height() as usize, vel_input.width() as usize), 0.0);

        let mut rng = rand::thread_rng();
        for y in 0..vx.dim().0 {
            for x in 0..vx.dim().1 {
                let px = vel_input.get_pixel(x as u32, y as u32);
                vx[(y, x)] = (px.data[0] as f64 / 255.0) * 200.0 - 100.0;
                vy[(y, x)] = (px.data[1] as f64 / 255.0) * 200.0 - 100.0;
            }
        }

        fwt_2d_separate_isotropic::<spline::Bior31, spline::Bior22>(vx.view_mut(), 3, temp.view_mut());
        fwt_2d_separate_isotropic::<spline::Bior22, spline::Bior31>(vy.view_mut(), 3, temp.view_mut());

        div_coeff_2d(3, vx.view(), vy.view(), div.view_mut(), n.view_mut());

        //
        let img_data = {
            let mut data = Vec::new();
            for y in 0..vx.dim().0 {
                for x in 0..vx.dim().1 {
                    let vx = &vx[(y, x)];
                    let vy = &vy[(y, x)];
                    data.push([
                        util::imgproc::transfer(vx, -10.0, 10.0),
                        util::imgproc::transfer(vy, -10.0, 10.0),
                        0,
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("vel_output.png"),
            &img_data,
            (vx.dim().1, vx.dim().0));

        let img_data = {
            let mut data = Vec::new();
            for y in 0..vx.dim().0 {
                for x in 0..vx.dim().1 {
                    let div = &div[(y, x)];
                    data.push([
                        util::imgproc::transfer(div, -10.0, 10.0),
                        util::imgproc::transfer(div, -10.0, 10.0),
                        util::imgproc::transfer(div, -10.0, 10.0),
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("div_output.png"),
            &img_data,
            (vx.dim().1, vx.dim().0));


        let img_data = {
            let mut data = Vec::new();
            for y in 0..vx.dim().0 {
                for x in 0..vx.dim().1 {
                    let n = &n[(y, x)];
                    data.push([
                        util::imgproc::transfer(n, -10.0, 10.0),
                        util::imgproc::transfer(n, -10.0, 10.0),
                        util::imgproc::transfer(n, -10.0, 10.0),
                    ]);
                }
            }
            data
        };

        util::png::export(
            format!("non-div_output.png"),
            &img_data,
            (vx.dim().1, vx.dim().0));
    }
}
