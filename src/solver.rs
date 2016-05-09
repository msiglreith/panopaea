
use std::cmp;
use cgmath;
use grid::*;
use math;
use stopwatch::{Stopwatch};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Cell {
    Solid,
    Fluid,
}

fn integrate_euler(pos: cgmath::Vector2<f64>, vel: cgmath::Vector2<f64>, dt: f64) -> cgmath::Vector2<f64> {
    pos + dt * vel
}

pub fn advect(dest: &mut Grid2d<f64>, quantity: &Grid2d<f64>, timestep: f64, vel: &MacGrid2d<f64>) {
    let q = &quantity;

    let (w, h) = q.dim;
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

pub fn apply_boundary_conditions(grid: &mut Grid2d<f64>, flags: &Grid2d<Cell>, flag: usize) {
    let (w, h) = grid.dim;
    for y in 0 .. h {
        grid[(0  , y)] = if flag == 1 { -grid[(1  , y)] } else { grid[(1  , y)] };
        grid[(w-1, y)] = if flag == 1 { -grid[(w-2, y)] } else { grid[(w-2, y)] };
    }

    for x in 0 .. w {
        grid[(x, 0  )] = if flag == 2 { -grid[(x, 1  )] } else { grid[(x, 1  )] };
        grid[(x, h-1)] = if flag == 2 { -grid[(x, h-2)] } else { grid[(x, h-2)] };
    }
}

pub fn advect_mac(dest: &mut MacGrid2d<f64>, quantity: &MacGrid2d<f64>, timestep: f64, vel: &MacGrid2d<f64>) {
    let q = &quantity;

    for y in 0 .. q.x.dim.1  {
        for x in 0 .. q.x.dim.0 {
            let vel = cgmath::vec2(vel.x[(x, y)],
                (vel.y[(cmp::min(x, vel.y.dim.0-1), y  )] +
                 vel.y[(cmp::min(x, vel.y.dim.0-1), y+1)] +
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

                let x0 = cmp::min(px + 0, q.x.dim.0 - 1); let y0 = cmp::min(py + 0, q.x.dim.1 - 1);
                let x1 = cmp::min(px + 1, q.x.dim.0 - 1); let y1 = cmp::min(py + 1, q.x.dim.1 - 1);

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

    for y in 0 .. q.y.dim.1  {
        for x in 0 .. q.y.dim.0 {
            let vel = cgmath::vec2(
                (vel.x[(x  , cmp::min(y, vel.x.dim.1-1))] +
                 vel.x[(x+1, cmp::min(y, vel.x.dim.1-1))] +
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

                let x0 = cmp::min(px + 0, q.y.dim.0 - 1); let y0 = cmp::min(py + 0, q.y.dim.1 - 1);
                let x1 = cmp::min(px + 1, q.y.dim.0 - 1); let y1 = cmp::min(py + 1, q.y.dim.1 - 1);

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

pub fn add_buoyancy(density: &Grid2d<f64>, vel: &mut MacGrid2d<f64>, timestep: f64) {
    let alpha = 1.0;
    let gravity = -1.4;

    for y in 1 .. vel.dim.1 - 0 {
        for x in 0 .. vel.dim.0 - 0 {
            let b = -alpha * gravity * timestep;
            vel.y[(x, y)] += b * 0.5 * (density[(x, y)] + density[(x, y - 1)]);
        }
    }
}

fn build_div(div: &mut Grid2d<f64>, vel: &mut MacGrid2d<f64>) {
    for y in 0 .. div.dim.1 - 0 {
        for x in 0 .. div.dim.0 - 0 {
            div[(x, y)] = -(vel.x[(x+1, y)] - vel.x[(x, y)] + vel.y[(x, y+1)] - vel.y[(x, y)]);
        }
    }
}

fn project_velocity(vel: &mut MacGrid2d<f64>, pressure: &Grid2d<f64>, timestep: f64) {
    let scale = timestep;
    for y in 0 .. vel.dim.1 {
        for x in 0 .. vel.dim.0 {
            if x < pressure.dim.0 { vel.x[(x, y)] -= scale*pressure[(x, y)]; }
            if x > 0 { vel.x[(x, y)] += scale*pressure[(x-1, y)]; }
            if y < pressure.dim.1 { vel.y[(x, y)] -= scale*pressure[(x, y)]; }
            if y > 0 { vel.y[(x, y)] += scale*pressure[(x, y-1)]; }
        }
    }

    for y in 0 .. vel.dim.1 {
        vel.x[(0, y)] = 0.0;
        vel.x[(vel.dim.0, y)] = 0.0;
    }

    for x in 0 .. vel.dim.0 {
        vel.y[(x, 0)] = 0.0;
        vel.y[(x, vel.dim.1)] = 0.0;
    }
}

pub fn project(
    pressure: &mut Grid2d<f64>,
    div: &mut Grid2d<f64>,
    vel: &mut MacGrid2d<f64>,
    timestep: f64,
    max_iterations: usize,
    threshold: f64,
) {
    pressure.fill_zero();
    build_div(div, vel);

    let sw_project_pressure = Stopwatch::start_new();
    let mut max_error: f64 = 0.0;
    'iter: for _ in 0..max_iterations {
        max_error = 0.0;
        for y in 0 .. pressure.dim.1 - 0 {
            for x in 0 .. pressure.dim.0 - 0 {
                pressure[(x, y)] = {
                    let mut diag = 0f64;
                    let mut kernel_sum = 0.0;

                    if x > 0 { diag += 1.0; kernel_sum += pressure[(x - 1, y)]; }
                    if x < pressure.dim.0 - 1 { diag += 1.0; kernel_sum += pressure[(x + 1, y)]; }
                    if y > 0 { diag += 1.0; kernel_sum += pressure[(x, y - 1)]; }
                    if y < pressure.dim.1 - 1 { diag += 1.0; kernel_sum += pressure[(x, y + 1)]; }

                    let new_p = (div[(x, y)] + kernel_sum * timestep) / (diag * timestep);
                    let old_p = pressure[(x, y)];

                    max_error = max_error.max(new_p - old_p);

                    new_p
                };
            }
        }

        if max_error < threshold {
            break 'iter;
        }
    }
    println!("      Project Pressure: {}ms", sw_project_pressure.elapsed_ms());
    println!("{:?}", max_error);

    project_velocity(vel, pressure, timestep);
}

fn apply_sparse_matrix(
    dest: &mut Grid2d<f64>,
    src: &Grid2d<f64>,
    diag: &Grid2d<f64>,
    plus_x: &Grid2d<f64>,
    plus_y: &Grid2d<f64>,
    timestep: f64,
) {
    let scale = timestep;
    for y in 0 .. src.dim.1 {
        for x in 0 .. src.dim.0 {
            dest[(x, y)] = {
                let mut b = diag[(x, y)] * src[(x, y)];
                if x > 0 { b += plus_x[(x-1, y)] * src[(x-1, y  )]; }
                if y > 0 { b += plus_y[(x, y-1)] * src[(x  , y-1)]; }
                if x < src.dim.0-1 { b += plus_x[(x, y)] * src[(x+1, y  )]; }
                if y < src.dim.1-1 { b += plus_y[(x, y)] * src[(x  , y+1)]; }
                b * scale
            };
        }
    }
}

pub fn build_sparse_matrix(
    diag: &mut Grid2d<f64>,
    plus_x: &mut Grid2d<f64>,
    plus_y: &mut Grid2d<f64>,
    timestep: f64,
) {
    diag.fill_zero();
    plus_x.fill_zero();
    plus_y.fill_zero();

    for y in 0..diag.dim.1 {
        for x in 0..diag.dim.0 {
            if x > 0 { diag[(x, y)] += 1.0; }
            if y > 0 { diag[(x, y)] += 1.0; }
            if x < diag.dim.0-1 { diag[(x, y)] += 1.0; plus_x[(x, y)] -= 1.0; }
            if y < diag.dim.1-1 { diag[(x, y)] += 1.0; plus_y[(x, y)] -= 1.0; }
        }
    }
}

pub fn project_cg(
    mut pressure: &mut Grid2d<f64>,
    div: &mut Grid2d<f64>,
    vel: &mut MacGrid2d<f64>,
    diag: &Grid2d<f64>,
    plus_x: &Grid2d<f64>,
    plus_y: &Grid2d<f64>,
    residual: &mut Grid2d<f64>,
    auxiliary: &mut Grid2d<f64>,
    mut search: &mut Grid2d<f64>,
    timestep: f64,
    max_iterations: usize,
    threshold: f64,
) {
    build_div(div, vel);
    
    // Conjugate gradient
    // Returns a pressure field to make the velocity field divergence-free

    // early out, nothing todo when the velocity field is already div-free
    if div.max_norm() < threshold {
        println!("div start norm: {:?}", div.max_norm());
        return;
    }

    // initial guess
    pressure.fill_zero();
    residual.copy(div);
    auxiliary.copy(residual);
    search.copy(auxiliary);

    let sw_project_pressure = Stopwatch::start_new();
    let mut residual_error = 0.0f64;
    let mut sigma = auxiliary.dot(residual);
    'iter: for i in 0..max_iterations {
        apply_sparse_matrix(auxiliary, search, diag, plus_x, plus_y, timestep);
        let alpha = sigma/auxiliary.dot(search);

        for i in 0..pressure.data.len() {
            pressure[i] += search[i] * alpha;
            residual[i] -= auxiliary[i] * alpha;
        }

        residual_error = residual.max_norm();
        if residual_error < threshold {
            println!("Iterations {:?}", i);
            break 'iter;
        }

        auxiliary.copy(residual);
        let sigma_new = auxiliary.dot(residual);
        let beta = sigma_new/sigma;

        for i in 0..search.data.len() {
            search[i] = auxiliary[i] + beta*search[i];
        }
        
        sigma = sigma_new;
    }

    println!("      Project Pressure: {}ms", sw_project_pressure.elapsed_ms());
    println!("{:?}", residual_error);

    project_velocity(vel, pressure, timestep);
}
