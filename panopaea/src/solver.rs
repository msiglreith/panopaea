
use na;

pub fn integrate_euler(pos: na::Vector2<f64>, vel: na::Vector2<f64>, dt: f64) -> na::Vector2<f64> {
    pos + dt * vel
}