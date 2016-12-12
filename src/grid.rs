
use ndarray::{ArrayBase, ArrayView, ArrayViewMut, Dimension, DataOwned};
use ndarray::{Array, Ix1, Ix2};
use cgmath::BaseFloat;

pub trait Grid<A: BaseFloat, D> {
    fn norm_max(&self) -> A;
    fn view_linear(&self) -> ArrayView<A, Ix1>;
    fn view_linear_mut(&mut self) -> ArrayViewMut<A, Ix1>;
}

impl<D: Dimension> Grid<f64, D> for ArrayBase<Vec<f64>, D> {
    fn norm_max(&self) -> f64 {
        self.iter().map(|x| f64::abs(*x)).fold(0.0, |max, x| f64::max(max, x))
    }

    fn view_linear(&self) -> ArrayView<f64, Ix1> {
        unsafe { ArrayView::<f64, Ix1>::from_shape_ptr(self.len(), self.as_ptr()) }
    }

    fn view_linear_mut(&mut self) -> ArrayViewMut<f64, Ix1> {
        unsafe { ArrayViewMut::<f64, Ix1>::from_shape_ptr(self.len(), self.as_mut_ptr()) }
    }
}

pub type Grid2D<T: BaseFloat> = Array<T, Ix2>;

pub struct MacGrid2D<T: BaseFloat> {
    pub x: Grid2D<T>,
    pub y: Grid2D<T>,
    pub dimension: (usize, usize),
}
