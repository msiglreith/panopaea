
extern crate panopaea;
extern crate panopaea_util as util;
extern crate ndarray;
extern crate stopwatch;

use panopaea::*;
use ndarray::Array2;
use stopwatch::Stopwatch;

fn main() {
    let domain = (128, 128);
    let mut solver = Solver::new();
    let mut density  = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut pressure = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut vel = MacGrid2D::new(
            domain,
            solver.allocate_grid_2d::<f64>((domain.0    , domain.1 + 1), 0.0),
            solver.allocate_grid_2d::<f64>((domain.0 + 1, domain.1    ), 0.0),
        );

    let mut vel_temp = MacGrid2D::new(
            domain,
            solver.allocate_grid_2d::<f64>((domain.0    , domain.1 + 1), 0.0),
            solver.allocate_grid_2d::<f64>((domain.0 + 1, domain.1    ), 0.0),
        );

    //
    let mut temp = solver.allocate_grid_2d::<f64>(domain, 0.0);

    //
    let mut diag   = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut plus_x = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut plus_y = solver.allocate_grid_2d::<f64>(domain, 0.0);

    // CG required helper grids
    let mut aux      = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut search   = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut residual = solver.allocate_grid_2d::<f64>(domain, 0.0);
    let mut cholesky  = solver.allocate_grid_2d::<f64>(domain, 0.0);

    let timestep = 0.05;
    let threshold = 0.1;

    for i in 0 .. 1000 {
        
        println!("{:?}", i);
        // inflow
        {
            let mut d = density.view_mut();
            for y in 5 .. 20 {
                for x in 54 .. 64 {
                    d[(y, x)] = 1.0;
                    vel.y[(y, x)] = 20.0;
                }
            }
        }

        // advection
        advect(&mut temp, &density, timestep, &vel);
        advect_mac(&mut vel_temp, &vel, timestep, &vel);

        density.assign(&temp);
        vel.assign(&vel_temp);

        // pressure projection
        cg::build_sparse_matrix(&mut diag, &mut plus_x, &mut plus_y, timestep);
        let precond = cg::ModIncCholesky::new(&mut cholesky, &diag, &plus_x, &plus_y);

        let sw = Stopwatch::start_new();
        cg::conjugate_gradient(&precond, &mut pressure, &mut temp, &mut vel,
            &diag, &plus_x, &plus_y,
            &mut residual, &mut aux, &mut search,
            timestep, 100, threshold);

        

        println!("{}ms", sw.elapsed_ms());

        project_velocity(&mut vel, &pressure, timestep);

        // debug output
        if i % 10 == 0 {
            let img_data = {
                let mut data = Vec::new();
                for y in 0 .. density.dim().0 {
                    for x in 0 .. density.dim().1 {
                        let val = &density[(y, x)];
                        data.push([
                            util::imgproc::transfer(val, 0.0, 1.0),
                            util::imgproc::transfer(val, 0.0, 1.0),
                            util::imgproc::transfer(val, 0.0, 1.0),
                        ]);
                    }
                }
                data
            };

            util::png::export(
                format!("output/density_{:?}.png", i),
                &img_data,
                density.dim());

            /*
            let img_vel_x_data = {
                let mut data = Vec::new();
                for y in 0 .. vel.x.dim().0 {
                    for x in 0 .. vel.x.dim().1 {
                        let val = &vel.x[(y, x)];
                        data.push([
                            util::imgproc::transfer(val, -10.0, 10.0),
                            util::imgproc::transfer(val, -10.0, 10.0),
                            util::imgproc::transfer(val, -10.0, 10.0),
                        ]);
                    }
                }
                data
            };

            util::png::export(
                format!("output/vel_x_{:?}.png", i),
                &img_vel_x_data,
                vel.x.dim());

            let img_vel_y_data = vel.y.iter().map(|val| 
                [
                    util::imgproc::transfer(val, -10.0, 10.0),
                    util::imgproc::transfer(val, -10.0, 10.0),
                    util::imgproc::transfer(val, -10.0, 10.0),
                ]).collect::<Vec<_>>();

            util::png::export(
                format!("output/vel_y_{:?}.png", i),
                &img_vel_y_data,
                vel.y.dim());

            */
        }
    }
}
