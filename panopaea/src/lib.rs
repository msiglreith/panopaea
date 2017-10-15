
extern crate nalgebra as na;
extern crate num;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate ndarray_parallel;
extern crate generic_array;
#[macro_use]
extern crate mopa;
extern crate rand;
extern crate rayon;
extern crate rustfft as fft;
extern crate typenum;
extern crate cgmath;
extern crate specs;
extern crate timely;

pub mod dec;
pub mod domain;
pub mod math;
pub mod ocean;
pub mod particle;
pub mod pbd;
pub mod pcg;
pub mod scene;
pub mod solver;
pub mod sparse;
pub mod sph;

pub use scene::*;
pub use solver::*;
