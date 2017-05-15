
//! Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)

use cgmath::MetricSpace;
use math::{Dim, Real};
use generic_array::ArrayLength;
use particle::{self, Particles};
use rayon::prelude::*;
use typenum::U2;

use super::grid::BoundedGrid;
use super::kernel::{self, Kernel};
use super::property::*;

pub fn init<T, N>(particles: &mut Particles)
    where T: Real + 'static,
          N: Dim<T>,
{
    particles.add_property::<Position<T, N>>();
    particles.add_property::<Velocity<T, N>>();
    particles.add_property::<Density<T>>();
    particles.add_property::<Mass<T>>();
}

pub fn compute_density<T>(kernel_size: T, grid: &BoundedGrid<T, U2>, particles: &mut Particles)
    where T: Real + 'static,
          // N: Dim<T> + Dim<usize> + Dim<(usize, usize)>,
{
    particles.run(|p| {
        let (mut density, position, mass) = (
            p.write_property::<Density<T>>().unwrap(),
            p.read_property::<Position<T, U2>>().unwrap(),
            p.read_property::<Mass<T>>().unwrap(),
        );

        let poly_6 = kernel::Poly6::new(kernel_size);

        density.par_iter_mut().enumerate()
            .zip(position.par_iter())
            .for_each(|((id, mut density), pos)| {
                let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
                let mut d = T::zero();
                println!("cell: {:?}", cell);
                for y in (cell.1.saturating_sub(1) .. cell.1.saturating_add(1)) {
                    for x in (cell.0.saturating_sub(1) .. cell.0.saturating_add(1)) {
                        let (start, end) = grid.get_range((x, y));
                        println!("range: {:?}", (start, end));
                        for p in start..end {
                            d += *mass[p] * poly_6.w(pos.distance(&position[p]));
                        }
                    }
                }

                *density = Density(d);
            });
    });
}
