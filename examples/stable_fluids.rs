
extern crate panopaea;
extern crate stopwatch;

use panopaea::*;
use stopwatch::{Stopwatch};

use std::io::Write;
use std::io::stdout;

fn main() {
    println!("Hello, world!");
    let domain_size = (128, 128);

    let mut flags: grid::Grid2d<Cell> = grid::Grid2d::new(domain_size, vec![Cell::Fluid; domain_size.0*domain_size.1]);
    let mut density: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut pressure: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut temp: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut vel: grid::MacGrid2d<f64> = grid::MacGrid2d::new(domain_size.0, domain_size.1);
    let mut vel_temp: grid::MacGrid2d<f64> = grid::MacGrid2d::new(domain_size.0, domain_size.1);

    let mut diag: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut plus_x: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut plus_y: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);

    let mut auxiliary: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut search: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);
    let mut residual: grid::Grid2d<f64> = grid::Grid2d::new(domain_size, vec![0.0; domain_size.0*domain_size.1]);

    // set wall boundaries
    {
        let (w, h) = flags.dim;
        for x in 0 .. w {
            flags[(x, 0)] = Cell::Solid;
            flags[(x, w-1)] = Cell::Solid;
        }

        for y in 0 .. h {
            flags[(0, y)] = Cell::Solid;
            flags[(h-1, y)] = Cell::Solid;
        }
    }

    let timestep = 0.05;
    let threshold = 0.1;

    for i in 0 .. 1200 {
        let sw = Stopwatch::start_new();
        let sw_inflow = Stopwatch::start_new();
        {
            let mut d = &mut density;
            for y in 5 .. 20 {
                for x in 60 .. 70 {
                    d[(x, y)] = 1.0;
                    vel.y[(x, y)] = 20.0;
                }
            }
        }
        println!("  Inflow: {}ms", sw_inflow.elapsed_ms());

        let sw_advect = Stopwatch::start_new();
        advect(&mut temp, &density, timestep, &vel);
        println!("  Advect (Density): {}ms", sw_advect.elapsed_ms());
        density.copy(&temp);

        let sw_advect_vel = Stopwatch::start_new();
        advect_mac(&mut vel_temp, &vel, timestep, &vel);
        println!("  Advect (Vel): {}ms", sw_advect_vel.elapsed_ms());
        vel.copy(&vel_temp);

        // add_buoyancy(&density, &mut vel, timestep);

        let sw_project = Stopwatch::start_new();

        build_sparse_matrix(&mut diag, &mut plus_x, &mut plus_y, timestep);
        project_cg(&mut pressure, &mut temp, &mut vel, &diag, &plus_x, &plus_y, &mut residual, &mut auxiliary, &mut search, timestep, 200, threshold);
        // project(&mut pressure, &mut temp, &mut vel, timestep, 200, threshold);
        println!("  Project: {}ms", sw_project.elapsed_ms());

        println!("{}ms", sw.elapsed_ms());

        if i % 10 != 0 {
            continue
        }

        println!("output vector fields");

        {
            let img_data = density.data.iter().map(|x| {
            [
                transfer(x, 0.0, 1.0),
                transfer(x, 0.0, 1.0),
                transfer(x, 0.0, 1.0),
            ]}).collect::<Vec<_>>();

            save_image_rgb(format!("output/density_{:?}.png", i), &img_data, density.dim.0, density.dim.1);
        }

        {
            let img_data = pressure.data.iter().map(|x| {
            [
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
            ]}).collect::<Vec<_>>();

            save_image_rgb(format!("output/pressure_{:?}.png", i), &img_data, pressure.dim.0, pressure.dim.1);
        }

        {
            let img_data = vel.x.data.iter().map(|x| {
            [
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
            ]}).collect::<Vec<_>>();

            save_image_rgb(format!("output/vel_x_{:?}.png", i), &img_data, vel.x.dim.0, vel.x.dim.1);
        }

        {
            let img_data = vel.y.data.iter().map(|x| {
            [
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
                transfer(x, -1.0, 1.0),
            ]}).collect::<Vec<_>>();

            save_image_rgb(format!("output/vel_y_{:?}.png", i), &img_data, vel.y.dim.0, vel.y.dim.1);
        }
    }
}
