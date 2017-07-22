
extern crate alga;
extern crate nalgebra as na;
extern crate num;
#[macro_use] extern crate ndarray;
#[macro_use] extern crate ndarray_parallel;
extern crate generic_array;
#[macro_use] extern crate mopa;
extern crate rayon;
extern crate typenum;
extern crate cgmath;
extern crate specs;
extern crate sprs;

pub mod cg;
pub mod dec;
pub mod domain;
pub mod grid;
pub mod math;
pub mod particle;
pub mod pbd;
pub mod pcg;
pub mod scene;
pub mod solver;
pub mod sparse;
pub mod sph;

pub use grid::*;
pub use scene::*;
pub use solver::*;
