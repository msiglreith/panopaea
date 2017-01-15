
extern crate cgmath;
extern crate ndarray;

use cgmath::BaseFloat;
use ndarray::{arr1, ArrayView, ArrayViewMut, Array1, Ix1};

fn fwt_1d(input: ArrayView<f64, Ix1>, outputs: &mut [ArrayViewMut<f64, Ix1>]) {
    outputs[0].assign(&input);

    for n in 1..outputs.len() {
        // TODO: generalize to all wavelets
        let (details, coarser) = outputs.split_at_mut(n);
        let detail = details.last_mut().unwrap();
        let coarse = coarser.first_mut().unwrap();

        for i in 0..coarse.len() {
            coarse[i] = (detail[2*i] + detail[2*i+1]) / 2.0;
        }

        for i in 0..detail.len() {
            detail[i] -= coarse[i/2];
        }
    }
}

fn main() {
    let input = arr1(&[12.0, 4.0, 6.0, 8.0, 4.0, 2.0, 5.0, 7.0]);
    let mut level0 = Array1::zeros(input.len());
    let mut level1 = Array1::zeros(input.len() / 2);

    fwt_1d(input.view(), &mut [level0.view_mut(), level1.view_mut()]);

    println!("{:?}", level0);
    println!("{:?}", level1);
}
