
extern crate alga;
extern crate nalgebra as na;
extern crate num;
#[macro_use]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate generic_array;

pub extern crate sprs as sparse;

pub mod cg;
pub mod dec;
pub mod grid;
pub mod math;
pub mod solver;
pub mod wavelet;

pub use grid::*;
pub use solver::*;
