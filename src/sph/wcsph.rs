
//! Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)

use cgmath::MetricSpace;
use math::{Dim, Real};
use particle::{Particles, Processor};
use rayon::prelude::*;
use typenum::U2;
use num::cast;

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
pub fn compute_density<T>(p: Processor, (kernel_size, grid): (T, &BoundedGrid<T, U2>))
    where T: Real + 'static,
          // N: Dim<T> + Dim<usize> + Dim<(usize, usize)>,
{
    let (mut density, position, masses) = (
        p.write_property::<Density<T>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<Mass<T>>(),
    );

    let poly_6 = kernel::Poly6::new(kernel_size);

    density.par_iter_mut().enumerate()
        .zip(masses.par_iter())
        .zip(position.par_iter())
        .for_each(|(((i, mut density), &mass), pos)| {
            let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };

            let mut d = mass * poly_6.w(T::zero());
            grid.for_each_neighbor(cell, 1, |p| {
                if p == i { return }
                d += masses[p] * poly_6.w(pos.distance(&position[p]));
            });

            *density = d;
        });
}

pub fn calculate_pressure<T>(p: Processor, (kernel_size, gas_constant, rest_density, grid): (T, T, T, &BoundedGrid<T, U2>))
    where T: Real + 'static,
{
    let (densities, positions, mut accels, masses) = (
        p.read_property::<Density<T>>(),
        p.read_property::<Position<T, U2>>(),
        p.write_property::<Acceleration<T, U2>>(),
        p.read_property::<Mass<T>>(),
    );

    let spiky = kernel::Poly6::new(kernel_size);

    densities.par_iter()
       .zip(positions.par_iter())
       .zip(accels.par_iter_mut())
       .for_each(|((&density, &pos), mut accel)| {
            let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
            let pressure_i = gas_constant * (density - rest_density);

            grid.for_each_neighbor(cell, 1, |p| {
                let pressure_j = gas_constant * (densities[p] - rest_density);
                let density_j = densities[p];
                let mass_j = masses[p];
                let two = cast::<f64, T>(2.0).unwrap();
                let r = pos - positions[p];
                *accel -= r * (mass_j * spiky.grad_w(pos.distance(&positions[p])) * (pressure_j + pressure_i) / (two * density_j * density));
            });
        });
}

pub fn calculate_viscosity<T>(p: Processor, (kernel_size,viscosity, grid): (T, T, &BoundedGrid<T, U2>))
    where T: Real + 'static,
{
    let (densities, positions, velocities, mut accels, masses) = (
        p.read_property::<Density<T>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<Velocity<T, U2>>(),
        p.write_property::<Acceleration<T, U2>>(),
        p.read_property::<Mass<T>>(),
    );

    let visc = kernel::Viscosity::new(kernel_size);

    densities.par_iter()
       .zip(positions.par_iter())
       .zip(velocities.par_iter())
       .zip(accels.par_iter_mut())
       .for_each(|(((&density, &pos), &vel), mut accel)| {
            let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };

            // TODO: skip own?
            grid.for_each_neighbor(cell, 1, |p| {
                let diff_vel = velocities[p] - vel;
                *accel += diff_vel * (viscosity * masses[p] / (density * densities[p]) * visc.laplace_w(pos.distance(&positions[p])));
            });

        });
}

pub fn integrate_explicit_euler<T>(p: Processor, timestep: T)
    where T: Real + 'static,
{
    let (mut positions, mut velocities, accels) = (
        p.write_property::<Position<T, U2>>(),
        p.write_property::<Velocity<T, U2>>(),
        p.read_property::<Acceleration<T, U2>>(),
    );

    positions.par_iter_mut()
        .zip(velocities.par_iter_mut())
        .zip(accels.par_iter())
        .for_each(|((mut pos, mut vel), &accel)| {
            *vel += accel * timestep;
            *pos += *vel * timestep;
        });
}