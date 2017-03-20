
use grid::{Grid, Grid2D, MacGrid2D};
use ndarray_parallel::prelude::*;

/// Conjugate gradient preconditioner.
///
/// TODO
pub trait Preconditioner {
    fn apply(&self, dst: &mut Grid2D<f64>, src: &Grid2D<f64>);
}

impl Preconditioner for () {
    fn apply(&self, dst: &mut Grid2D<f64>, src: &Grid2D<f64>) { dst.assign(src); }
}

pub struct ModIncCholesky<'a> {
    precond: &'a mut Grid2D<f64>,
    plus_x: &'a Grid2D<f64>,
    plus_y: &'a Grid2D<f64>,
}

impl<'a> ModIncCholesky<'a> {
    pub fn new(precond: &'a mut Grid2D<f64>, diag: &Grid2D<f64>, plus_x: &'a Grid2D<f64>, plus_y: &'a Grid2D<f64>) -> Self {
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
    fn apply(&self, dst: &mut Grid2D<f64>, src: &Grid2D<f64>) {
        unsafe {
            let (h, w) = src.dim();
            let mut idx = 0;
            for y in 0 .. h {
                for x in 0 .. w {
                    idx += 1;
                    let mut t = src.at(idx);
                    if x > 0 { t -= self.plus_x.at(idx-1) * self.precond.at(idx-1) * dst.at(idx-1) };
                    if y > 0 { t -= self.plus_y.at(idx-w) * self.precond.at(idx-w) * dst.at(idx-w) };
                    dst[(y, x)] = t * self.precond.at(idx);
                }
            }

            let mut idx = 0;
            for y in (0 .. h).rev() {
                for x in (0 .. w).rev() {
                    idx += 1;
                    let mut t = dst.at(idx);
                    if x < w - 1 { t -= self.plus_x.at(idx) * self.precond.at(idx) * dst.at(idx+1) };
                    if y < h - 1 { t -= self.plus_y.at(idx) * self.precond.at(idx) * dst.at(idx+w) };
                    dst[(y, x)] = t * self.precond.at(idx);
                }
            }
        }
    }
}

fn build_div(div: &mut Grid2D<f64>, vel: &mut MacGrid2D<f64>) {
    div.indexed_iter_mut().map(|((y, x), div)| {
        *div = -(vel.x[(y, x+1)] - vel.x[(y, x)] + vel.y[(y+1, x)] - vel.y[(y, x)]);
    });
}

fn apply_sparse_matrix(
    dest: &mut Grid2D<f64>,
    src: &Grid2D<f64>,
    diag: &Grid2D<f64>,
    plus_x: &Grid2D<f64>,
    plus_y: &Grid2D<f64>,
    timestep: f64,
) {
    unsafe {
        let scale = timestep;
        let (h, w) = src.dim();
        dest.indexed_iter_mut().map(|((y, x), dst)| {
            let mut idx = y*w + x;
            *dst = {
                let mut b = diag.at(idx) * src.at(idx);
                if x > 0 { b += plus_x.at(idx-1) * src.at(idx-1); }
                if y > 0 { b += plus_y.at(idx-w) * src.at(idx-w); }
                if x < src.dim().1-1 { b += plus_x.at(idx) * src.at(idx+1); }
                if y < src.dim().0-1 { b += plus_y.at(idx) * src.at(idx+w); }
                b * scale
            };
        });
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

pub fn conjugate_gradient<P: Preconditioner>(
    preconditioner: &P,
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
