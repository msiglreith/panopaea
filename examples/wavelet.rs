
extern crate cgmath;
#[macro_use]
extern crate ndarray;
extern crate image;
extern crate panopaea_util as util;
extern crate generic_array;

use image::GenericImage;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1};
use generic_array::ArrayLength;
use generic_array::typenum::{U2};

pub trait Wavelet<T> {
    type N: ArrayLength<usize>;
    fn coeff_up_low() -> &'static [T];
    fn coeff_up_high() -> &'static [T];
    fn coeff_down_low() -> &'static [T];
    fn coeff_down_high() -> &'static [T];
}

// Haar wavelet
static HAAR_UP_LOW:  &[f64; 2] = &[0.7071067811865476,  0.7071067811865476];
static HAAR_UP_HIGH: &[f64; 2] = &[0.7071067811865476, -0.7071067811865476];
static HAAR_DOWN_LOW : &[f64; 2] = &[ 0.7071067811865476, 0.7071067811865476];
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


#[inline]
fn down_convolution(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = output.len() / 2;

    let low_pass = Haar::coeff_down_low();
    let high_pass = Haar::coeff_down_high();

    // low-pass
    for i in 0..half_size {
        output[i] =
            input[2*i  ] * low_pass[0] +
            input[2*i+1] * low_pass[1];
    }

    // high pass
    for i in 0..half_size {
        output[i+half_size] =
            input[2*i  ] * high_pass[0] +
            input[2*i+1] * high_pass[1];
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
    }
}
