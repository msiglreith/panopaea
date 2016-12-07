
use ndarray::{ArrayBase, Dimension, DataOwned};
use ndarray::{Array, Ix2};
use cgmath::BaseFloat;

pub trait Grid<S: DataOwned, D> {

}

impl<S: DataOwned, D> Grid<S, D> for ArrayBase<S, D> {

}

pub type Grid2D<T: BaseFloat> = Array<T, Ix2>;

pub struct MacGrid2D<T: BaseFloat> {
    pub x: Grid2D<T>,
    pub y: Grid2D<T>,
    pub dimension: (usize, usize),
}
