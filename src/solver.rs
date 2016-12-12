
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

    pub fn allocate_grid_2d<T: BaseFloat>(&self, size: (usize, usize)) -> Grid2D<T> {
        Array2::<T>::zeros(size)
    }
}

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

pub fn advect_mac(dest: &mut MacGrid2D<f64>, quantity: &MacGrid2D<f64>, timestep: f64, vel: &MacGrid2D<f64>) {
    let q = &quantity;

    for y in 0 .. q.x.dim().1  {
        for x in 0 .. q.x.dim().0 {
            let vel = cgmath::vec2(vel.x[(x, y)],
                (vel.y[(cmp::min(x, vel.y.dim().0-1), y  )] +
                 vel.y[(cmp::min(x, vel.y.dim().0-1), y+1)] +
                 vel.y[(x.saturating_sub(1), y  )] +
                 vel.y[(x.saturating_sub(1), y+1)])
                 / 4.0);

            let pos = cgmath::vec2(
                (x as f64 + 0.0),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dest.x[(x, y)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.0).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.5).floor().max(0.0) as usize;

                let x0 = cmp::min(px + 0, q.x.dim().0 - 1); let y0 = cmp::min(py + 0, q.x.dim().1 - 1);
                let x1 = cmp::min(px + 1, q.x.dim().0 - 1); let y1 = cmp::min(py + 1, q.x.dim().1 - 1);

                // // println!("    temp - amount");
                let s = (pos_prev.x - 0.0 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.5 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    q.x[(x0, y0)], q.x[(x1, y0)],
                    q.x[(x0, y1)], q.x[(x1, y1)],
                    s, t
                )
            };
        }
    }

    for y in 0 .. q.y.dim().1  {
        for x in 0 .. q.y.dim().0 {
            let vel = cgmath::vec2(
                (vel.x[(x  , cmp::min(y, vel.x.dim().1-1))] +
                 vel.x[(x+1, cmp::min(y, vel.x.dim().1-1))] +
                 vel.x[(x  , y.saturating_sub(1))] +
                 vel.x[(x+1, y.saturating_sub(1))]
                ) / 4.0,
                vel.y[(x, y)]);

            let pos = cgmath::vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.0)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dest.y[(x, y)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.5).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.0).floor().max(0.0) as usize;

                let x0 = cmp::min(px + 0, q.y.dim().0 - 1); let y0 = cmp::min(py + 0, q.y.dim().1 - 1);
                let x1 = cmp::min(px + 1, q.y.dim().0 - 1); let y1 = cmp::min(py + 1, q.y.dim().1 - 1);

                let s = (pos_prev.x - 0.5 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.0 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    q.y[(x0, y0)], q.y[(x1, y0)],
                    q.y[(x0, y1)], q.y[(x1, y1)],
                    s, t
                )
            };
        }
    }
}

fn build_div(div: &mut Grid2D<f64>, vel: &mut MacGrid2D<f64>) {
    for y in 0 .. div.dim().1 {
        for x in 0 .. div.dim().0 {
            div[(x, y)] = -(vel.x[(x+1, y)] - vel.x[(x, y)] + vel.y[(x, y+1)] - vel.y[(x, y)]);
        }
    }
}

fn project_velocity(vel: &mut MacGrid2D<f64>, pressure: &Grid2D<f64>, timestep: f64) {
    let scale = timestep;
    for y in 0 .. vel.dimension.1 {
        for x in 0 .. vel.dimension.0 {
            if x < pressure.dim().0 { vel.x[(x, y)] -= scale*pressure[(x, y)]; }
            if x > 0 { vel.x[(x, y)] += scale*pressure[(x-1, y)]; }
            if y < pressure.dim().1 { vel.y[(x, y)] -= scale*pressure[(x, y)]; }
            if y > 0 { vel.y[(x, y)] += scale*pressure[(x, y-1)]; }
        }
    }

    for y in 0 .. vel.dimension.1 {
        vel.x[(0, y)] = 0.0;
        vel.x[(vel.dimension.0, y)] = 0.0;
    }

    for x in 0 .. vel.dimension.0 {
        vel.y[(x, 0)] = 0.0;
        vel.y[(x, vel.dimension.1)] = 0.0;
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
    for y in 0 .. src.dim().1 {
        for x in 0 .. src.dim().0 {
            dest[(x, y)] = {
                let mut b = diag[(x, y)] * src[(x, y)];
                if x > 0 { b += plus_x[(x-1, y)] * src[(x-1, y  )]; }
                if y > 0 { b += plus_y[(x, y-1)] * src[(x  , y-1)]; }
                if x < src.dim().0-1 { b += plus_x[(x, y)] * src[(x+1, y  )]; }
                if y < src.dim().1-1 { b += plus_y[(x, y)] * src[(x  , y+1)]; }
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

    for y in 0..diag.dim().1 {
        for x in 0..diag.dim().0 {
            if x > 0 { diag[(x, y)] += 1.0; }
            if y > 0 { diag[(x, y)] += 1.0; }
            if x < diag.dim().0-1 { diag[(x, y)] += 1.0; plus_x[(x, y)] -= 1.0; }
            if y < diag.dim().1-1 { diag[(x, y)] += 1.0; plus_y[(x, y)] -= 1.0; }
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
        let mut sigma = auxiliary_grid.view_linear().dot(&auxiliary_grid.view_linear());

        'iter: for i in 0..max_iterations {
            apply_sparse_matrix(auxiliary_grid, search_grid, diag, plus_x, plus_y, timestep);
            let alpha = sigma/auxiliary_grid.view_linear().dot(&search_grid.view_linear());
            

            pressure.scaled_add( alpha, search_grid);
            residual.scaled_add(-alpha, auxiliary_grid);

            residual_error = residual.norm_max();
            if residual_error < threshold {
                println!("Iterations {:?}", i);
                break 'iter;
            }

            auxiliary_grid.assign(residual);
            let auxiliary = auxiliary_grid.view_linear();
            let sigma_new = auxiliary.dot(&residual.view_linear());
            let beta = sigma_new/sigma;

            let mut search = search_grid.view_linear_mut();

            for i in 0..search.len() {
                search[i] = auxiliary[i] + beta*search[i];
            }

            sigma = sigma_new;
        }
 
    }
    

    project_velocity(vel, pressure, timestep);
}
