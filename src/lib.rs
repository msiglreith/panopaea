
extern crate cgmath;
extern crate num;
extern crate ndarray;
extern crate ndarray_parallel;
extern crate generic_array;

pub mod cg;
pub mod grid;
pub mod math;
pub mod solver;
pub mod wavelet;

pub use grid::*;
pub use solver::*;
