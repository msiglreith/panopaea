
//! Wavelet and Divergence-free wavelet transforms
//!
//! > NOTE: These were earlier tests, mostly abandonded and might not be fully working!
use ndarray::{ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1, Ix2, Ixs};
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

// Array sampler
pub trait Sampler<T> {
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

pub struct MirrorSampler;
impl Sampler<f64> for MirrorSampler {
    fn fetch(input: ArrayView<f64, Ix1>, idx: Ixs) -> f64 {
        let len = input.len() as isize;
        if idx < 0 {
            input[(-idx) as usize]
        } else if idx >= len {
            input[(len - 1 - (idx % len)) as usize]
        } else {
            input[idx as usize]
        }
    }
}

#[inline]
pub fn down_convolution<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
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
            MirrorSampler::fetch(input, input_src) * low_pass[filter_src] + acc
        });
    }

    // high pass
    for i in 0..half_size {
        output[(i+half_size) as usize] = (0..filter_size).fold(0.0, |acc, f| {
            let input_src = 2*i+f-filter_half+1;
            let filter_src = (filter_size-1-f) as usize;
            MirrorSampler::fetch(input, input_src) * high_pass[filter_src] + acc
        });
    }
}

#[inline]
pub fn up_convolution<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
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
                let idx = 2 * i + f - filter_half + 1;
                if idx < 0 {
                    (idx + output.len() as isize) % output.len() as isize
                } else {
                    idx % output.len() as isize
                }
            };
            let filter_src = f as usize;
            output[idx_out as usize] +=
                MirrorSampler::fetch(input, i) * low_pass[filter_src] +
                MirrorSampler::fetch(input, i+half_size) * high_pass[filter_src];
        }
    }
}

pub fn fwt_1d<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix1>, levels: usize, mut temp: ArrayViewMut<f64, Ix1>) {
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

pub fn ifwt_1d<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix1>, levels: usize, mut temp: ArrayViewMut<f64, Ix1>) {
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

pub fn fwt_2d_isotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
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

pub fn ifwt_2d_isotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
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

pub fn fwt_2d_separate_isotropic<Wx, Wy>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>)
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

pub fn ifwt_2d_separate_isotropic<Wx, Wy>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>)
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

        // x direction
        for i in 0..level_size.0 {
            up_convolution::<Wx>(
                src.subview(Axis(0), i as usize),
                dest.subview_mut(Axis(0), i as usize));
        }

        src.assign(&dest);

        // y direction
        for i in 0..level_size.1 {
            up_convolution::<Wy>(
                src.subview(Axis(1), i as usize),
                dest.subview_mut(Axis(1), i as usize));
        }
    }
}

pub fn fwt_2d_anisotropic<W: Wavelet<f64>>(mut output: ArrayViewMut<f64, Ix2>, levels: usize, mut temp: ArrayViewMut<f64, Ix2>) {
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

pub fn div_coeff_2d(levels: usize,
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
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

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

pub fn inv_div_coeff_2d(levels: usize,
    src_div: ArrayView<f64, Ix2>, src_n: ArrayView<f64, Ix2>,
    mut dst_vx: ArrayViewMut<f64, Ix2>, mut dst_vy: ArrayViewMut<f64, Ix2>)
{
    // all arrays should have the same dimension
    debug_assert!(src_div.dim() == src_n.dim(),
        "Input arrays differ in dimension (div: {:?}, n: {:?})",
        src_div.dim(), src_n.dim());
    debug_assert!(src_div.dim() == dst_vx.dim(),
        "src (div) and dst (vx) differ in dimension (src: {:?}, dst: {:?})",
        src_div.dim(), dst_vx.dim());
    debug_assert!(src_n.dim() == dst_vy.dim(),
        "src (n) and dst (vy) differ in dimension (src: {:?}, dst: {:?})",
        src_n.dim(), dst_vy.dim());


    let level_pow = 2usize.pow(levels as u32);
    let mut level_size = src_div.dim();
    level_size = (level_size.0 / level_pow, level_size.1 / level_pow);

    for n in 0 .. levels {
        let coarse_slice = s![..2 * level_size.0 as isize, ..2 * level_size.1 as isize];
        let src_div = src_div.slice(coarse_slice);
        let src_n = src_n.slice(coarse_slice);
        let mut dst_vx = dst_vx.slice_mut(coarse_slice);
        let mut dst_vy = dst_vy.slice_mut(coarse_slice);

        // split into wavelet components
        let (div_00, div_01, div_10, div_11) = {
            let (low_y, high_y) = src_div.view().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (n_00, n_01, n_10, n_11) = {
            let (low_y, high_y) = src_n.view().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut vx_00, mut vx_01, mut vx_10, mut vx_11) = {
            let (low_y, high_y) = dst_vx.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        let (mut vy_00, mut vy_01, mut vy_10, mut vy_11) = {
            let (low_y, high_y) = dst_vy.view_mut().split_at(Axis(0), level_size.0);
            let (lylx, lyhx) = low_y.split_at(Axis(1), level_size.1);
            let (hylx, hyhx) = high_y.split_at(Axis(1), level_size.1);
            (lylx, lyhx, hylx, hyhx)
        };

        for y in 0..level_size.0 {
            for x in 0..level_size.1 {
                vx_01[(y, x)] = n_01[(y, x)] - (div_01[(y.saturating_sub(1), x)] - div_01[(y, x)]) / 4.0;
                vx_10[(y, x)] = div_10[(y, x)];
                vx_11[(y, x)] = div_11[(y, x)] + n_11[(y, x)];

                vy_01[(y, x)] = div_01[(y, x)];
                vy_10[(y, x)] = n_10[(y, x)] - (div_10[(y, x.saturating_sub(1))] - div_10[(y, x)]) / 4.0;
                vy_11[(y, x)] = n_11[(y, x)] - div_11[(y, x)];
            }
        }

        if n == 0 {
            vx_00.assign(&div_00);
            vy_00.assign(&n_00);
        }

        level_size = (level_size.0 * 2, level_size.1 * 2);
    }
}


pub fn fwt_1d_border<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, levels: usize) -> (Array1<f64>, Vec<Array1<f64>>) {
    let mut details = Vec::new();
    let mut coarse = Array1::zeros(input.len());
    coarse.assign(&input);

    for _ in 0..levels {
        let filter_size = W::filter_length();
        let border_width = W::border_width();
        let band_size = coarse.len() / 2 + 2 * border_width as usize;
        let mut detail = Array1::zeros(band_size);
        let mut down = Array1::zeros(band_size);

        down_convolution_border::<W>(coarse.view(), down.view_mut(), detail.view_mut());

        details.push(detail);
        coarse = down;
    }

    (coarse, details)
}

pub fn ifwt_1d_border<W: Wavelet<f64>>(&(ref coarse, ref details): &(Array1<f64>, Vec<Array1<f64>>)) -> Array1<f64> {
    let mut coarse = coarse.clone();
    let filter_size = W::filter_length();

    for detail in details.iter().rev() {
        let input_len = coarse.len() + detail.len();
        let border_width = W::border_width();
        let mut up = Array1::zeros(input_len - 4*border_width as usize);
        up_convolution_border::<W>(coarse.view(), detail.view(), up.view_mut());
        coarse = up;
    }

    coarse
}

pub fn down_convolution_border<W: Wavelet<f64>>(input: ArrayView<f64, Ix1>, mut coarse: ArrayViewMut<f64, Ix1>, mut detail: ArrayViewMut<f64, Ix1>) {
    let filter_size = W::filter_length();
    let filter_half = filter_size / 2;
    let border_width = W::border_width();
    let half_size = detail.len() as isize;

    let low_pass = W::coeff_down_low();
    let high_pass = W::coeff_down_high();

    // low-pass
    for i in 0..half_size {
        coarse[i as usize] = (0..filter_size).fold(0.0, |acc, f| {
            let input_src = 2*(i-border_width)+f-filter_half+1;
            let filter_src = (filter_size-1-f) as usize;
            ZeroSampler::fetch(input, input_src) * low_pass[filter_src] + acc
        });
    }

    // high pass
    for i in 0..half_size {
        detail[i as usize] = (0..filter_size).fold(0.0, |acc, f| {
            let input_src = 2*(i-border_width)+f-filter_half+1;
            let filter_src = (filter_size-1-f) as usize;
            ZeroSampler::fetch(input, input_src) * high_pass[filter_src] + acc
        });
    }
}

pub fn up_convolution_border<W: Wavelet<f64>>(coarse: ArrayView<f64, Ix1>, detail: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let filter_size = W::filter_length();
    let input_len = coarse.len() + detail.len();

    let in_half_size = (input_len / 2) as isize;
    let filter_half = filter_size / 2;
    let border_width = W::border_width();
    let low_pass = W::coeff_up_low();
    let high_pass = W::coeff_up_high();

    output.fill(0.0);
    for i in 0..in_half_size {
        for f in 0..filter_size {
            let idx_out = {
                let idx = 2 * i + f - filter_half + 1;
                if idx < 0 {
                    (idx + input_len as isize) % input_len as isize
                } else {
                    idx % input_len as isize
                }
            };

            let idx_out = idx_out - 2 * border_width;
            if idx_out < 0 || idx_out >= output.len() as isize {
                continue;
            }

            let filter_src = f as usize;
            output[idx_out as usize] +=
                ZeroSampler::fetch(coarse, i) * low_pass[filter_src] +
                ZeroSampler::fetch(detail, i) * high_pass[filter_src];
        }
    }
}

pub fn fwt_2d_separate_isotropic_border<Wx, Wy>(input: ArrayView<f64, Ix2>, levels: usize) -> (Array2<f64>, Vec<[Array2<f64>; 3]>)
    where Wx: Wavelet<f64>, Wy: Wavelet<f64>
{
    let mut details = Vec::new();
    let mut coarse = Array2::zeros(input.dim());
    coarse.assign(&input);

    for n in 0..levels {
        let filter_size_x = Wx::filter_length();
        let border_width_x = Wx::border_width();
        let filter_size_y = Wy::filter_length();
        let border_width_y = Wy::border_width();
        let band_size = (
            coarse.dim().0 / 2 + 2 * border_width_y as usize,
            coarse.dim().1 / 2 + 2 * border_width_x as usize);

        let mut down_lx = Array2::zeros((coarse.dim().0, band_size.1));
        let mut detail_hx = Array2::zeros((coarse.dim().0, band_size.1));
        let mut detail_lyhx = Array2::zeros(band_size);
        let mut detail_hylx = Array2::zeros(band_size);
        let mut detail_hyhx = Array2::zeros(band_size);
        let mut down = Array2::zeros(band_size);

        // x direction
        for i in 0..coarse.dim().0 {
            down_convolution_border::<Wx>(
                coarse.subview(Axis(0), i as usize),
                down_lx.subview_mut(Axis(0), i as usize),
                detail_hx.subview_mut(Axis(0), i as usize));
        }

        // y direction
        for i in 0..band_size.1 {
            down_convolution_border::<Wy>(
                down_lx.subview(Axis(1), i as usize),
                down.subview_mut(Axis(1), i as usize),
                detail_hylx.subview_mut(Axis(1), i as usize));
            down_convolution_border::<Wy>(
                detail_hx.subview(Axis(1), i as usize),
                detail_lyhx.subview_mut(Axis(1), i as usize),
                detail_hyhx.subview_mut(Axis(1), i as usize));
        }

        coarse = down;
        details.push([detail_lyhx, detail_hylx, detail_hyhx]);
    }

    (coarse, details)
}

pub fn ifwt_2d_separate_isotropic_border<Wx, Wy>(&(ref coarse, ref details): &WaveletTransform2d) -> Array2<f64>
    where Wx: Wavelet<f64>, Wy: Wavelet<f64>
{
    let mut coarse = coarse.clone();
    let filter_size_x = Wx::filter_length();
    let filter_size_y = Wy::filter_length();

    for (n, detail) in details.iter().enumerate().rev() {
        let border_width_x = Wx::border_width();
        let border_width_y = Wy::border_width();
        let mut up = Array2::zeros((
            2 * coarse.dim().0 - 4 * border_width_y as usize,
            2 * coarse.dim().1 - 4 * border_width_x as usize));

        let mut up_lx = Array2::zeros((2 * coarse.dim().0 - 4 * border_width_y as usize, coarse.dim().1));
        let mut detail_hx = Array2::zeros((2 * coarse.dim().0 - 4 * border_width_y as usize, coarse.dim().1));

        let detail_lyhx = &detail[0];
        let detail_hylx = &detail[1];
        let detail_hyhx = &detail[2];

        // y direction
        for i in 0..coarse.dim().1 {
            up_convolution_border::<Wy>(
                coarse.subview(Axis(1), i as usize),
                detail_hylx.subview(Axis(1), i as usize),
                up_lx.subview_mut(Axis(1), i as usize));

            up_convolution_border::<Wy>(
                detail_lyhx.subview(Axis(1), i as usize),
                detail_hyhx.subview(Axis(1), i as usize),
                detail_hx.subview_mut(Axis(1), i as usize));
        }

        // x direction
        for i in 0..up.dim().0 {
            up_convolution_border::<Wx>(
                up_lx.subview(Axis(0), i as usize),
                detail_hx.subview(Axis(0), i as usize),
                up.subview_mut(Axis(0), i as usize));
        }

        coarse = up;
    }

    coarse
}

pub type WaveletTransform2d = (Array2<f64>, Vec<[Array2<f64>; 3]>);

pub fn div_coeff_2d_border(vx: &WaveletTransform2d, vy: &WaveletTransform2d) -> (WaveletTransform2d, WaveletTransform2d)
{
    let div_coarse = vx.0.clone();
    let n_coarse = vy.0.clone();

    let mut div_details = Vec::new();
    let mut n_details = Vec::new();

    for (detail_vx, detail_vy) in vx.1.iter().zip(vy.1.iter()) {
        let band_size = detail_vx[0].dim();

        // split into wavelet components
        let (vx_01, vx_10, vx_11) = (&detail_vx[0], &detail_vx[1], &detail_vx[2]);
        let (vy_01, vy_10, vy_11) = (&detail_vy[0], &detail_vy[1], &detail_vy[2]);

        let (mut div_01, mut div_10, mut div_11) = (Array2::zeros(band_size), Array2::zeros(band_size), Array2::zeros(band_size));
        let (mut n_01, mut n_10, mut n_11) = (Array2::zeros(band_size), Array2::zeros(band_size), Array2::zeros(band_size));

        for y in 0..band_size.0 {
            for x in 0..band_size.1 {
                div_01[(y, x)] = vy_01[(y, x)];
                div_10[(y, x)] = vx_10[(y, x)];
                div_11[(y, x)] = (vx_11[(y, x)] - vy_11[(y, x)]) / 2.0;

                n_01[(y, x)] = vx_01[(y, x)] + (vy_01[(y, x)] - vy_01[(y.saturating_sub(1), x)]) / 4.0; // TODO
                n_10[(y, x)] = vy_10[(y, x)] + (vx_10[(y, x)] - vx_10[(y, x.saturating_sub(1))]) / 4.0; // TODO
                n_11[(y, x)] = (vx_11[(y, x)] + vy_11[(y, x)]) / 2.0;
            }
        }

        div_details.push([div_01, div_10, div_11]);
        n_details.push([n_01, n_10, n_11]);
    }

    ((div_coarse, div_details), (n_coarse, n_details))
}
