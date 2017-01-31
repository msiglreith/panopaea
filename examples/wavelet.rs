
extern crate cgmath;
#[macro_use]
extern crate ndarray;
extern crate image;
extern crate panopaea_util as util;

use image::GenericImage;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Array2, Axis, Ix1};

// Haar reconstruction
// low-pass:  0.7071067811865476 0.7071067811865476
// high-pass: 0.7071067811865476 0.7071067811865476

#[inline]
fn down_convolution(input: ArrayView<f64, Ix1>, mut output: ArrayViewMut<f64, Ix1>) {
    let half_size = output.len() / 2;

    // TODO: generalize to all wavelets
    // TODO: non-pow2 arrays

    // low-pass
    for i in 0..half_size {
        output[i] =
            input[2*i  ] * 0.7071067811865476 +
            input[2*i+1] * 0.7071067811865476;
    }

    // high pass
    for i in 0..half_size {
        output[i+half_size] =
           -input[2*i  ] * 0.7071067811865476 +
            input[2*i+1] * 0.7071067811865476;
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

        for x in 0..lena.width() as usize {
            let col = (0..lena.height() as usize).into_iter().map(|y| {
                output[(y, x)]
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
