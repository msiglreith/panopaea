
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
    particles.add_property::<Acceleration<T, N>>();
    particles.add_property::<Density<T>>();
    particles.add_property::<Mass<T>>();
}

/// Compute particle density approximation based on the smoothing kernel.
///
/// Ref: [MDM03] Eq. 3
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

        density.par_iter_mut()
            .zip(position.par_iter())
            .for_each(|(mut density, pos)| {
                let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
                let mut d = T::zero();
                for y in (cell.1.saturating_sub(1) .. cell.1.saturating_add(2)) {
                    for x in (cell.0.saturating_sub(1) .. cell.0.saturating_add(2)) {
                        let (start, end) = grid.get_range((x, y));
                        for p in start..end {
                            d += *mass[p] * poly_6.w(pos.distance(&position[p]));
                        }
                    }
                }

                *density = Density(d);
            });
    });
}

pub fn calculate_pressure<T>(kernel_size: T, gas_constant: T, rest_density: T, grid: &BoundedGrid<T, U2>, particles: &mut Particles)
    where T: Real + 'static,
{
    particles.run(|p| {
        let (density, position, mut accel, mass) = (
            p.read_property::<Density<T>>().unwrap(),
            p.read_property::<Position<T, U2>>().unwrap(),
            p.write_property::<Acceleration<T, U2>>().unwrap(),
            p.read_property::<Mass<T>>().unwrap(),
        );

        density.par_iter()
           .zip(position.par_iter())
           .zip(accel.par_iter_mut())
           .for_each(|((density, pos), mut accel)| {
                let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
                for y in (cell.1.saturating_sub(1) .. cell.1.saturating_add(2)) {
                    for x in (cell.0.saturating_sub(1) .. cell.0.saturating_add(2)) {
                        let (start, end) = grid.get_range((x, y));
                        for p in start..end {
                            accel.0 += *mass[p];
                        }
                    }
                }
            });
    });
}