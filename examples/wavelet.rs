
extern crate cgmath;
#[macro_use]
extern crate ndarray;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Ix1};

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
        // TODO: generalize to all wavelets
        // TODO: non-pow2 arrays
        let half_size = level_size / 2;

        for i in 0..half_size {
            output[i] = 
                temp[2*i  ] * 0.7071067811865476 +
                temp[2*i+1] * 0.7071067811865476;
        }

        for i in 0..half_size {
            output[i+half_size] =
               -temp[2*i  ] * 0.7071067811865476 +
                temp[2*i+1] * 0.7071067811865476;
        }
    }
}


// Haar reconstruction
// low-pass:  0.7071067811865476 0.7071067811865476
// high-pass: 0.7071067811865476 0.7071067811865476

fn main() {
    let input = arr1(&[12.0, 4.0, 6.0, 8.0, 4.0, 2.0, 5.0, 7.0]);
    let mut decomposition = Array1::zeros(input.len());
    let mut temp = Array1::zeros(input.len());

    fwt_1d(input.view(), decomposition.view_mut(), 1, temp.view_mut());

    println!("{:?}", decomposition);
}
