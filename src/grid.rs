
use ndarray::{ArrayBase, ArrayView, ArrayViewMut, Dimension, DataOwned};
use ndarray::{Array, Ix1, Ix2, LinalgScalar};
use std::f64;

pub trait Grid<A: LinalgScalar, D> {
    fn norm_max(&self) -> A;
    fn view_linear(&self) -> ArrayView<A, Ix1>;
    fn view_linear_mut(&mut self) -> ArrayViewMut<A, Ix1>;
    fn dot_linear<Rhs: Grid<A, D>>(&self, rhs: &Rhs) -> A {
        self.view_linear().dot(&rhs.view_linear())
    }
    fn min_max(&self) -> (f64, f64);

    unsafe fn at(&self, usize) -> A;
    unsafe fn at_mut(&mut self, usize) -> &mut A;
}

impl<D: Dimension> Grid<f64, D> for ArrayBase<Vec<f64>, D> {
    fn norm_max(&self) -> f64 {
        let mut max = 0.0;
        for x in self.view_linear() {
            // Using max or abs is incredibly slow!
            max = if *x > max { *x } else if -x > max { -x } else { max };
        }
        max
    }

    fn view_linear(&self) -> ArrayView<f64, Ix1> {
        unsafe { ArrayView::<f64, Ix1>::from_shape_ptr(self.len(), self.as_ptr()) }
    }

    fn view_linear_mut(&mut self) -> ArrayViewMut<f64, Ix1> {
        unsafe { ArrayViewMut::<f64, Ix1>::from_shape_ptr(self.len(), self.as_mut_ptr()) }
    }

    fn min_max(&self) -> (f64, f64) {
        let min = self.iter().fold(f64::INFINITY, |min, &x| f64::min(min, x));
        let max = self.iter().fold(f64::NEG_INFINITY, |max, &x| f64::max(max, x));
        (min, max)
    }

    unsafe fn at(&self, idx: usize) -> f64 {
        *self.as_ptr().offset(idx as isize)
    }

    unsafe fn at_mut(&mut self, idx: usize) -> &mut f64 {
        &mut *self.as_mut_ptr().offset(idx as isize)
    }
}

pub type Grid2D<T: LinalgScalar> = Array<T, Ix2>;

pub struct MacGrid2D<T: LinalgScalar> {
    pub x: Grid2D<T>,
    pub y: Grid2D<T>,
    pub dimension: (usize, usize),
}

impl<T: LinalgScalar> MacGrid2D<T> {
    pub fn new(dim: (usize, usize), grid_x: Grid2D<T>, grid_y: Grid2D<T>) -> Self {
        // TODO[prio:high]: check subgrid dimensions
        MacGrid2D {
            x: grid_x,
            y: grid_y,
            dimension: dim,
        }
    }

    pub fn assign(&mut self, rhs: &MacGrid2D<T>) {
        // TODO[prio:high]: check dimensions
        self.x.assign(&rhs.x);
        self.y.assign(&rhs.y);
    }
}