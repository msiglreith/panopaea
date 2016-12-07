
use grid::{Grid2D, MacGrid2D};
use math;

use ndarray::{Array, Ix2};
use cgmath::{self, BaseFloat};

pub fn integrate_euler(pos: cgmath::Vector2<f64>, vel: cgmath::Vector2<f64>, dt: f64) -> cgmath::Vector2<f64> {
    pos + dt * vel
}

pub fn advect<T: BaseFloat>(dest: &mut Grid2D<f64>, quantity: &Grid2D<f64>, timestep: f64, vel: &MacGrid2D<f64>) {
    let q = &quantity;

    let (w, h) = q.dim();
    for y in 0 .. h {
        for x in 0 .. w {
            let vel_center = cgmath::vec2(
                (vel.x[(x, y)] + vel.x[(x + 1, y    )]) / 2.0,
                (vel.y[(x, y)] + vel.y[(x    , y + 1)]) / 2.0
            );

            let pos = cgmath::vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel_center, -timestep);

            dest[(x, y)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.5).max(0.0).min(w as f64 - 1.00001);
                let py = (pos_prev.y - 0.5).max(0.0).min(h as f64 - 1.00001);

                let ix = px.floor() as usize;
                let iy = py.floor() as usize;

                let u = px - ix as f64;
                let v = py - iy as f64;

                math::bilinear(
                    q[(ix, iy  )], q[(ix+1, iy  )],
                    q[(ix, iy+1)], q[(ix+1, iy+1)],
                    u, v
                )
            };
        }
    }
}
