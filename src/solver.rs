
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

fn build_div(div: &mut Grid2D<f64>, vel: &mut MacGrid2D<f64>) {
    for y in 0 .. div.dim().0 {
        for x in 0 .. div.dim().1 {
            div[(y, x)] = -(vel.x[(y, x+1)] - vel.x[(y, x)] + vel.y[(y+1, x)] - vel.y[(y, x)]);
        }
    }
}

fn project_velocity(vel: &mut MacGrid2D<f64>, pressure: &Grid2D<f64>, timestep: f64) {
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


fn apply_sparse_matrix(
    dest: &mut Grid2D<f64>,
    src: &Grid2D<f64>,
    diag: &Grid2D<f64>,
    plus_x: &Grid2D<f64>,
    plus_y: &Grid2D<f64>,
    timestep: f64,
) {
    let scale = timestep;
    for y in 0 .. src.dim().0 {
        for x in 0 .. src.dim().1 {
            dest[(y, x)] = {
                let mut b = diag[(y, x)] * src[(y, x)];
                if x > 0 { b += plus_x[(y  , x-1)] * src[(y  , x-1)]; }
                if y > 0 { b += plus_y[(y-1, x  )] * src[(y-1, x  )]; }
                if x < src.dim().1-1 { b += plus_x[(y, x)] * src[(y  , x+1)]; }
                if y < src.dim().0-1 { b += plus_y[(y, x)] * src[(y+1, x  )]; }
                b * scale
            };
        }
    }
}

pub fn build_sparse_matrix(
    diag: &mut Grid2D<f64>,
    plus_x: &mut Grid2D<f64>,
    plus_y: &mut Grid2D<f64>,
    timestep: f64,
) {
    diag.fill(0.0);
    plus_x.fill(0.0);
    plus_y.fill(0.0);

    for y in 0..diag.dim().0 {
        for x in 0..diag.dim().1 {
            if x > 0 { diag[(y, x)] += 1.0; }
            if y > 0 { diag[(y, x)] += 1.0; }
            if x < diag.dim().1-1 { diag[(y, x)] += 1.0; plus_x[(y, x)] -= 1.0; }
            if y < diag.dim().0-1 { diag[(y, x)] += 1.0; plus_y[(y, x)] -= 1.0; }
        }
    }
}

pub fn project_cg(
    mut pressure: &mut Grid2D<f64>,
    div: &mut Grid2D<f64>,
    vel: &mut MacGrid2D<f64>,
    diag: &Grid2D<f64>,
    plus_x: &Grid2D<f64>,
    plus_y: &Grid2D<f64>,
    residual: &mut Grid2D<f64>,
    auxiliary_grid: &mut Grid2D<f64>,
    mut search_grid: &mut Grid2D<f64>,
    timestep: f64,
    max_iterations: usize,
    threshold: f64,
) {
    build_div(div, vel);
    
    // Conjugate gradient
    // Returns a pressure field to make the velocity field divergence-free

    // early out, nothing todo when the velocity field is already div-free
    if div.norm_max() < threshold {
        println!("div start norm: {:?}", div.norm_max());
        return;
    }

    // initial guess
    pressure.fill(0.0);
    residual.assign(div);
    auxiliary_grid.assign(residual);
    search_grid.assign(auxiliary_grid);

    {
        let mut residual_error = 0.0f64;
        let mut sigma = auxiliary_grid.dot_linear(auxiliary_grid);

        'iter: for i in 0..max_iterations {
            apply_sparse_matrix(auxiliary_grid, search_grid, diag, plus_x, plus_y, timestep);
            let alpha = sigma/auxiliary_grid.dot_linear(search_grid);
            
            pressure.scaled_add( alpha, search_grid);
            residual.scaled_add(-alpha, auxiliary_grid);

            residual_error = residual.norm_max();
            if residual_error < threshold {
                println!("Iterations {:?}", i);
                break 'iter;
            }

            auxiliary_grid.assign(residual);
            
            let sigma_new = auxiliary_grid.dot_linear(residual);
            let beta = sigma_new/sigma;

            let mut search = search_grid.view_linear_mut();
            let auxiliary = auxiliary_grid.view_linear();

            for i in 0..search.len() {
                search[i] = auxiliary[i] + beta*search[i];
            }

            sigma = sigma_new;
        }
    }

    project_velocity(vel, pressure, timestep);
}
