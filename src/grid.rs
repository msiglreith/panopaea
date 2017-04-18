
use math::LinearView;
use ndarray::{ArrayBase, ArrayView, ArrayViewMut, Dimension, DataOwned};
use ndarray::{Array, Ix1, Ix2, LinalgScalar};
use std::f64;

pub trait Grid<A: LinalgScalar, D> : LinearView<Elem = A> {
    fn min_max(&self) -> (f64, f64);
}

impl<D: Dimension> Grid<f64, D> for ArrayBase<Vec<f64>, D> {
    fn min_max(&self) -> (f64, f64) {
        let min = self.iter().fold(f64::INFINITY, |min, &x| f64::min(min, x));
        let max = self.iter().fold(f64::NEG_INFINITY, |max, &x| f64::max(max, x));
        (min, max)
    }
}

pub type Grid2d<T: LinalgScalar> = Array<T, Ix2>;

pub struct MacGrid2d<T: LinalgScalar> {
    pub x: Grid2d<T>,
    pub y: Grid2d<T>,
    pub dimension: (usize, usize),
}

impl<T: LinalgScalar> MacGrid2d<T> {
    pub fn new(dim: (usize, usize), grid_x: Grid2d<T>, grid_y: Grid2d<T>) -> Self {
        // TODO[prio:high]: check subgrid dimensions
        MacGrid2d {
            x: grid_x,
            y: grid_y,
            dimension: dim,
        }
    }

    pub fn assign(&mut self, rhs: &MacGrid2d<T>) {
        // TODO[prio:high]: check dimensions
        self.x.assign(&rhs.x);
        self.y.assign(&rhs.y);
    }
}
