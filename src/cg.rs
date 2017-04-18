
use grid::{Grid, Grid2d, MacGrid2d};
use math::{LinearView, LinearViewReal, Real};
use ndarray;
use ndarray_parallel::prelude::*;

/// Conjugate gradient preconditioner.
///
/// TODO
pub trait Preconditioner {
    fn apply(&self, dst: &mut Grid2d<f64>, src: &Grid2d<f64>);
}

impl Preconditioner for () {
    fn apply(&self, dst: &mut Grid2d<f64>, src: &Grid2d<f64>) { dst.assign(src); }
}

pub struct ModIncCholesky<'a> {
    precond: &'a mut Grid2d<f64>,
    plus_x: &'a Grid2d<f64>,
    plus_y: &'a Grid2d<f64>,
}

impl<'a> ModIncCholesky<'a> {
    pub fn new(precond: &'a mut Grid2d<f64>, diag: &Grid2d<f64>, plus_x: &'a Grid2d<f64>, plus_y: &'a Grid2d<f64>) -> Self {
        // Tuning constant
        let tau = 0.97;
        // Safety constant
        let sigma = 0.25;

        // Not needed as preconditioner will always be prefilled
        // precond.fill(0.0);

        // Solve Lq = r
        for y in 0 .. precond.dim().0 {
            for x in 0 .. precond.dim().1 {
                let mut e = diag[(y, x)];

                if x > 0 {
                    let ax = plus_x[(y, x-1)] * precond[(y, x-1)];
                    let ay = plus_y[(y, x-1)] * precond[(y, x-1)];
                    e = e - ax*ax - tau * ax*ay;
                }

                if y > 0 {
                    let ax = plus_x[(y-1, x)] * precond[(y-1, x)];
                    let ay = plus_y[(y-1, x)] * precond[(y-1, x)];
                    e = e - ay*ay - tau * ax*ay;
                }

                if e < sigma * diag[(y, x)] {
                    e = diag[(y, x)];
                }

                precond[(y, x)] = 1.0 / e.sqrt();
            }
        }

        ModIncCholesky {
            precond: precond,
            plus_x: plus_x,
            plus_y: plus_y,
        }
    }
}

impl<'a> Preconditioner for ModIncCholesky<'a> {
    fn apply(&self, dst: &mut Grid2d<f64>, src: &Grid2d<f64>) {
        let (h, w) = dst.dim();

        let mut dst = dst.as_slice_mut().unwrap();
        let src = src.as_slice().unwrap();
        let plus_x = self.plus_x.as_slice().unwrap();
        let plus_y = self.plus_y.as_slice().unwrap();
        let precond = self.precond.as_slice().unwrap();

        {
            let mut idx = 0;
            for y in 0 .. h {
                for x in 0 .. w {
                    let mut t = src[idx];
                    if x > 0 { t -= plus_x[idx-1] * precond[idx-1] * dst[idx-1] };
                    if y > 0 { t -= plus_y[idx-w] * precond[idx-w] * dst[idx-w] };
                    dst[idx] = t * precond[idx];
                    idx += 1;
                }
            }

            // TODO: This is quite slow!
            for y in (0 .. h).rev() {
                for x in (0 .. w).rev() {
                    let mut idx = x + y*w; // TODO:
                    let mut t = dst[idx];
                    if x < w - 1 { t -= plus_x[idx] * precond[idx] * dst[idx+1] };
                    if y < h - 1 { t -= plus_y[idx] * precond[idx] * dst[idx+w] };
                    dst[idx] = t * precond[idx];
                }
            }
        }
    }
}

fn build_div(div: &mut Grid2d<f64>, vel: &mut MacGrid2d<f64>) {
    /*
    for y in 0 .. div.dim().0 {
        for x in 0 .. div.dim().1 {
            div[(y, x)] = -(vel.x[(y, x+1)] - vel.x[(y, x)] + vel.y[(y+1, x)] - vel.y[(y, x)]);
        }
    }
    */

    ndarray::Zip::from(div)
        .and(vel.x.slice(s![.., 1..]))
        .and(vel.x.slice(s![.., ..-1]))
        .and(vel.y.slice(s![1.., ..]))
        .and(vel.y.slice(s![..-1, ..]))
        .apply(|div, &vx1, &vx2, &vy1, &vy2| {
            *div = -(vx1 - vx2 + vy1 - vy2);
        });
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
    let (h, w) = src.dim();
    let mut idx = 0;

    let mut dst = dest.as_slice_mut().unwrap();
    let src = src.as_slice().unwrap();
    let diag = diag.as_slice().unwrap();
    let plus_x = plus_x.as_slice().unwrap();
    let plus_y = plus_y.as_slice().unwrap();

    for y in 0 .. h {
        for x in 0 .. w {
            let mut b = diag[idx] * src[idx];
            if x > 0 { b += plus_x[idx-1] * src[idx-1]; }
            if y > 0 { b += plus_y[idx-w] * src[idx-w]; }
            if x < w-1 { b += plus_x[idx] * src[idx+1]; }
            if y < h-1 { b += plus_y[idx] * src[idx+w]; }
        
            dst[idx] = b * scale;

            idx += 1;
        }
    }
}

pub fn build_sparse_matrix(
    diag: &mut Grid2d<f64>,
    plus_x: &mut Grid2d<f64>,
    plus_y: &mut Grid2d<f64>,
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

pub fn conjugate_gradient<P: Preconditioner>(
    preconditioner: &P,
    mut pressure: &mut Grid2d<f64>,
    div: &mut Grid2d<f64>,
    vel: &mut MacGrid2d<f64>,
    diag: &Grid2d<f64>,
    plus_x: &Grid2d<f64>,
    plus_y: &Grid2d<f64>,
    residual: &mut Grid2d<f64>,
    auxiliary_grid: &mut Grid2d<f64>,
    mut search_grid: &mut Grid2d<f64>,
    timestep: f64,
    max_iterations: usize,
    threshold: f64,
) {
    build_div(div, vel);
    
    // Conjugate gradient
    // Returns a pressure field to make the velocity field divergence-free

    // initial guess
    pressure.fill(0.0);

    // early out, nothing todo when the velocity field is already div-free
    if div.norm_max() < threshold {
        println!("div start norm: {:?}", div.norm_max());
        return;
    }

    residual.assign(div);
    preconditioner.apply(auxiliary_grid, residual);
    search_grid.assign(auxiliary_grid);

    {
        let mut residual_error = 0.0f64;
        let mut sigma = auxiliary_grid.dot_linear(residual);

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

            preconditioner.apply(auxiliary_grid, residual);
            
            let sigma_new = auxiliary_grid.dot_linear(residual);
            let beta = sigma_new/sigma;

            let mut search = search_grid.view_linear_mut();
            let auxiliary = auxiliary_grid.view_linear();

            for s in 0..search.len() {
                search[s] = auxiliary[s] + beta*search[s];
            }

            sigma = sigma_new;
        }
    }
}
