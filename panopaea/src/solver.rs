
use cgmath::BaseFloat as Real;
use na;
use ndarray::{Array2};
use std::cmp;

use math::{self, vec2};

pub fn integrate_euler(pos: na::Vector2<f64>, vel: na::Vector2<f64>, dt: f64) -> na::Vector2<f64> {
    pos + dt * vel
}