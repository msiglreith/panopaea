
use grid::{Grid, Grid2D, MacGrid2D};
use math;

use ndarray::{Array, Array2, Ix2};
use cgmath::{self, BaseFloat};

use std::cmp;

pub struct Solver {

}

impl Solver {
    pub fn new() -> Solver {
        Solver {

        }
    }

    pub fn allocate_grid_2d<T: BaseFloat>(&self, size: (usize, usize), val: T) -> Grid2D<T> {
        Array2::<T>::from_elem(size, val)
    }
}

pub fn integrate_euler(pos: cgmath::Vector2<f64>, vel: cgmath::Vector2<f64>, dt: f64) -> cgmath::Vector2<f64> {
    pos + dt * vel
}

pub fn advect(dest: &mut Grid2D<f64>, quantity: &Grid2D<f64>, timestep: f64, vel: &MacGrid2D<f64>) {
    let q = &quantity;

    let (h, w) = q.dim();
    for y in 0 .. h {
        for x in 0 .. w {
            let vel_center = cgmath::vec2(
                (vel.x[(y, x)] + vel.x[(y, x+1)]) / 2.0,
                (vel.y[(y, x)] + vel.y[(y+1, x)]) / 2.0
            );

            let pos = cgmath::vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel_center, -timestep);

            dest[(y, x)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.5).max(0.0).min(w as f64 - 1.00001);
                let py = (pos_prev.y - 0.5).max(0.0).min(h as f64 - 1.00001);

                let ix = px.floor() as usize;
                let iy = py.floor() as usize;

                let u = px - ix as f64;
                let v = py - iy as f64;

                math::bilinear(
                    q[(iy  , ix)], q[(iy  , ix+1)],
                    q[(iy+1, ix)], q[(iy+1, ix+1)],
                    u, v
                )
            };
        }
    }
}

pub fn advect_mac(dest: &mut MacGrid2D<f64>, quantity: &MacGrid2D<f64>, timestep: f64, vel: &MacGrid2D<f64>) {
    let q = &quantity;

    for y in 0 .. q.x.dim().0  {
        for x in 0 .. q.x.dim().1 {
            let vel = cgmath::vec2(vel.x[(y, x)],
                (vel.y[(y  , cmp::min(x, vel.y.dim().1-1))] +
                 vel.y[(y+1, cmp::min(x, vel.y.dim().1-1))] +
                 vel.y[(y  , x.saturating_sub(1))] +
                 vel.y[(y+1, x.saturating_sub(1))])
                 / 4.0);

            let pos = cgmath::vec2(
                (x as f64 + 0.0),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dest.x[(y, x)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.0).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.5).floor().max(0.0) as usize;

                let x0 = cmp::min(px + 0, q.x.dim().1 - 1); let y0 = cmp::min(py + 0, q.x.dim().0 - 1);
                let x1 = cmp::min(px + 1, q.x.dim().1 - 1); let y1 = cmp::min(py + 1, q.x.dim().0 - 1);

                // // println!("    temp - amount");
                let s = (pos_prev.x - 0.0 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.5 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    q.x[(y0, x0)], q.x[(y0, x1)],
                    q.x[(y1, x0)], q.x[(y1, x1)],
                    s, t
                )
            };
        }
    }

    for y in 0 .. q.y.dim().0  {
        for x in 0 .. q.y.dim().1 {
            let vel = cgmath::vec2(
                (vel.x[(cmp::min(y, vel.x.dim().0-1), x  )] +
                 vel.x[(cmp::min(y, vel.x.dim().0-1), x+1)] +
                 vel.x[(y.saturating_sub(1), x  )] +
                 vel.x[(y.saturating_sub(1), x+1)]
                ) / 4.0,
                vel.y[(y, x)]);

            let pos = cgmath::vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.0)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dest.y[(y, x)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.5).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.0).floor().max(0.0) as usize;

                let x0 = cmp::min(px    , q.y.dim().1 - 1); let y0 = cmp::min(py    , q.y.dim().0 - 1);
                let x1 = cmp::min(px + 1, q.y.dim().1 - 1); let y1 = cmp::min(py + 1, q.y.dim().0 - 1);

                let s = (pos_prev.x - 0.5 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.0 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    q.y[(y0, x0)], q.y[(y0, x1)],
                    q.y[(y1, x0)], q.y[(y1, x1)],
                    s, t
                )
            };
        }
    }
}

pub fn project_velocity(vel: &mut MacGrid2D<f64>, pressure: &Grid2D<f64>, timestep: f64) {
    let scale = timestep;
    for y in 0 .. vel.dimension.0 {
        for x in 0 .. vel.dimension.1 {
            if x < pressure.dim().1 { vel.x[(y, x)] -= scale*pressure[(y, x)]; }
            if x > 0 { vel.x[(y, x)] += scale*pressure[(y, x-1)]; }
            if y < pressure.dim().0 { vel.y[(y, x)] -= scale*pressure[(y, x)]; }
            if y > 0 { vel.y[(y, x)] += scale*pressure[(y-1, x)]; }
        }
    }

    for y in 0 .. vel.dimension.0 {
        vel.x[(y, 0)] = 0.0;
        vel.x[(y, vel.dimension.1)] = 0.0;
    }

    for x in 0 .. vel.dimension.1 {
        vel.y[(0, x)] = 0.0;
        vel.y[(vel.dimension.0, x)] = 0.0;
    }
}
