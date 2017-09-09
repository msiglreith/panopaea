
extern crate panopaea;
extern crate panopaea_utils as util;
extern crate nalgebra as na;
extern crate ndarray;
extern crate stopwatch;

use ndarray::Array2;
use panopaea::pcg;
use panopaea::math::{self, vec2};
use panopaea::domain::Grid2d;
use panopaea::dec::grid::{Simplex0, Simplex1, Simplex2};
use panopaea::dec::manifold::Manifold2d;
use panopaea::math::LinearView;
use std::cmp;
use stopwatch::Stopwatch;

fn grid2d_simplex1(grid: &Grid2d) -> Simplex1<f64> {
    <Grid2d as Manifold2d<f64>>::new_simplex_1(&grid)
}

fn grid2d_simplex2(grid: &Grid2d) -> Simplex2<f64> {
    <Grid2d as Manifold2d<f64>>::new_simplex_2(&grid)
}

fn main() {
    let grid = Grid2d::new((128, 128));

    let mut vel = grid2d_simplex1(&grid);
    let mut pressure = grid2d_simplex2(&grid);
    let mut density = grid2d_simplex2(&grid);

    let mut vel_temp = grid2d_simplex1(&grid);
    let mut vel_primal_temp = grid2d_simplex1(&grid);
    let mut temp = grid2d_simplex2(&grid);
    let mut pressure_temp = grid2d_simplex2(&grid);

    // conjugate gradient
    let mut auxiliary = grid2d_simplex2(&grid);
    let mut residual = grid2d_simplex2(&grid);
    let mut search = grid2d_simplex2(&grid);

    let timestep = 0.05;
    let threshold = 0.1;

    for i in 0 .. 1000 {
        // inflow
        {
            let mut d = density.view_mut();
            let (mut vy, mut vx) = vel.split_mut();
            for y in 5 .. 20 {
                for x in 54 .. 64 {
                    d[(y, x)] = 1.0;
                    vy[(y, x)] = 20.0;
                }
            }
        }

        advect(&mut temp, &density, timestep, &vel);
        advect_mac(&mut vel_temp, &vel, timestep, &vel);

        density.assign(&temp);
        vel.view_linear_mut().assign(&vel_temp.view_linear());

        vel_temp.view_linear_mut().fill(0.0);
        temp.view_linear_mut().fill(0.0);

        // calculate -div
        grid.hodge_1_dual(&mut vel_temp, &vel);
        {
            let (mut vy, mut vx) = vel_temp.split_mut();
            for y in 70 .. 80 {
                for x in 50 .. 70 {
                    vy[(y, x)] = 0.0;
                    vx[(y, x)] = 0.0;
                }
            }
        }

        grid.derivative_1_primal(&mut temp, &mut vel_temp);
        for x in temp.iter_mut() {
            *x = -*x;
        }

        // println!("div {:#?}", &temp);

        let sw = Stopwatch::start_new();

        vel_temp.view_linear_mut().fill(0.0);

        pcg::precond_conjugate_gradient(
            &(),
            &mut pressure,
            &temp,
            100,
            threshold,
            &mut residual,
            &mut auxiliary,
            &mut search,
            |mut laplacian, p| {
                grid.hodge_2_primal(&mut pressure_temp, &p);
                grid.derivative_0_dual(&mut vel_temp, &pressure_temp);

                {
                    let (mut vy, mut vx) = vel_temp.split_mut();
                    for y in 70 .. 80 {
                        for x in 50 .. 70 {
                            vy[(y, x)] = 0.0;
                            vx[(y, x)] = 0.0;
                        }
                    }
                }

                grid.hodge_1_dual(&mut vel_primal_temp, &vel_temp);
                grid.derivative_1_primal(&mut laplacian, &vel_primal_temp);
                for x in laplacian.iter_mut() {
                    *x = *x * timestep;
                }
            });

        println!("{} ms", sw.elapsed_ms());

        // project velocity
        grid.hodge_2_primal(&mut pressure_temp, &pressure);
        grid.derivative_0_dual(&mut vel_temp, &pressure_temp);
        vel.view_linear_mut().scaled_add(timestep, &vel_temp.view_linear());

        {
            let (mut vy, mut vx) = vel.split_mut();
            let max_x = vx.dim().1 - 1;
            let max_y = vy.dim().0 - 1;
            for y in 0 .. vx.dim().0 {
                vx[(y, 0)] = 0.0;
                vx[(y, max_x)] = 0.0;
            }

            for x in 0 .. vy.dim().1 {
                vy[(0, x)] = 0.0;
                vy[(max_y, x)] = 0.0;
            }
        }

        if i % 10 == 0 {
            let (img_data, dim) = {
                let mut data = Vec::new();
                // let (density, _) = vel.split();
                for y in 0 .. density.dim().0 {
                    for x in 0 .. density.dim().1 {
                        let val = &density[(y, x)];
                        data.push([
                            util::imgproc::transfer(val, -2.0, 2.0),
                            util::imgproc::transfer(val, -2.0, 2.0),
                            util::imgproc::transfer(val, -2.0, 2.0),
                        ]);
                    }
                }
                (data, (density.dim().1, density.dim().0))
            };

            util::png::export(
                format!("dec_fluid/density_{:?}.png", i),
                &img_data,
                dim);
        }

    }
}

pub fn integrate_euler(pos: na::Vector2<f64>, vel: na::Vector2<f64>, dt: f64) -> na::Vector2<f64> {
    pos + dt * vel
}

pub fn advect(dst: &mut Simplex2<f64>, src: &Simplex2<f64>, timestep: f64, vel: &Simplex1<f64>) {
    let q = &src;
    let (vy, vx) = vel.split();

    let (h, w) = q.dim();
    for y in 0 .. h {
        for x in 0 .. w {
            let vel_center = vec2(
                (vx[(y, x)] + vx[(y, x+1)]) / 2.0,
                (vy[(y, x)] + vy[(y+1, x)]) / 2.0
            );

            let pos = vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel_center, -timestep);

            dst[(y, x)] = {
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

pub fn advect_mac(dst: &mut Simplex1<f64>, src: &Simplex1<f64>, timestep: f64, vel: &Simplex1<f64>) {
    let (qy, qx) = src.split();
    let (mut dy, mut dx) = dst.split_mut();
    let (vy, vx) = vel.split();

    for y in 0 .. qx.dim().0  {
        for x in 0 .. qx.dim().1 {
            let vel = vec2(vx[(y, x)],
                (vy[(y  , cmp::min(x, vy.dim().1-1))] +
                 vy[(y+1, cmp::min(x, vy.dim().1-1))] +
                 vy[(y  , x.saturating_sub(1))] +
                 vy[(y+1, x.saturating_sub(1))])
                 / 4.0);

            let pos = vec2(
                (x as f64 + 0.0),
                (y as f64 + 0.5)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dx[(y, x)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.0).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.5).floor().max(0.0) as usize;

                let x0 = cmp::min(px + 0, qx.dim().1 - 1); let y0 = cmp::min(py + 0, qx.dim().0 - 1);
                let x1 = cmp::min(px + 1, qx.dim().1 - 1); let y1 = cmp::min(py + 1, qx.dim().0 - 1);

                // // println!("    temp - amount");
                let s = (pos_prev.x - 0.0 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.5 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    qx[(y0, x0)], qx[(y0, x1)],
                    qx[(y1, x0)], qx[(y1, x1)],
                    s, t
                )
            };
        }
    }

    for y in 0 .. qy.dim().0  {
        for x in 0 .. qy.dim().1 {
            let vel = vec2(
                (vx[(cmp::min(y, vx.dim().0-1), x  )] +
                 vx[(cmp::min(y, vx.dim().0-1), x+1)] +
                 vx[(y.saturating_sub(1), x  )] +
                 vx[(y.saturating_sub(1), x+1)]
                ) / 4.0,
                vy[(y, x)]);

            let pos = vec2(
                (x as f64 + 0.5),
                (y as f64 + 0.0)
            );

            let pos_prev = integrate_euler(pos, vel, -timestep);

            dy[(y, x)] = {
                // interpolate bilinear
                let px = (pos_prev.x - 0.5).floor().max(0.0) as usize;
                let py = (pos_prev.y - 0.0).floor().max(0.0) as usize;

                let x0 = cmp::min(px    , qy.dim().1 - 1); let y0 = cmp::min(py    , qy.dim().0 - 1);
                let x1 = cmp::min(px + 1, qy.dim().1 - 1); let y1 = cmp::min(py + 1, qy.dim().0 - 1);

                let s = (pos_prev.x - 0.5 - px as f64).min(1.0).max(0.0);
                let t = (pos_prev.y - 0.0 - py as f64).min(1.0).max(0.0);

                math::bilinear(
                    qy[(y0, x0)], qy[(y0, x1)],
                    qy[(y1, x0)], qy[(y1, x1)],
                    s, t
                )
            };
        }
    }
}
