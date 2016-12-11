
extern crate panopaea;
extern crate ndarray;

use panopaea::*;
use ndarray::Array2;

fn main() {
    let domain = (256, 256);
    let mut solver = Solver::new();
    let mut density = solver.allocate_grid_2d::<f64>(domain);

}
