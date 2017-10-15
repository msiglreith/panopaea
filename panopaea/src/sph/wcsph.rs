
//! Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)

use cgmath::MetricSpace;
use generic_array::ArrayLength;
use math::{Dim, Real};
use particle::{Particles, Processor};
use num::cast;

use super::kernel::{self, Kernel};
use super::neighbor::NeighborSearch;
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
pub fn compute_density<T, N, NS>(p: &Processor, (kernel_size, search): (T, &NS))
where
    T: Real + 'static,
    N: Dim<T>,
    <N as ArrayLength<T>>::ArrayType: Copy,
    NS: NeighborSearch<T, N>,
{
    let (density, position, masses) = (
        p.write_property::<Density<T>>(),
        p.read_property::<Position<T, N>>(),
        p.read_property::<Mass<T>>(),
    );

    let poly_6 = kernel::Poly6::new(kernel_size);

    par_azip!(
        index i,
        mut density (density),
        mass (masses),
        pos (position),
    in {
        let cell = if let Some(cell) = search.get_cell(&pos) { cell } else { return };

        let mut d = mass * poly_6.w(T::zero());
        search.for_each_neighbor(cell, 1, |p| {
            if p == i { return }
            d += masses[p] * poly_6.w(pos.distance(position[p]));
        });

        *density = d;
    });
}

pub fn calculate_pressure<T, N, NS>(p: &Processor, (kernel_size, gas_constant, rest_density, serch): (T, T, T, &NS))
where
    T: Real + 'static,
    N: Dim<T>,
    <N as ArrayLength<T>>::ArrayType: Copy,
    NS: NeighborSearch<T, N>,
{
    let (densities, positions, accels, masses) = (
        p.read_property::<Density<T>>(),
        p.read_property::<Position<T, N>>(),
        p.write_property::<Acceleration<T, N>>(),
        p.read_property::<Mass<T>>(),
    );

    let spiky = kernel::Poly6::new(kernel_size);

    par_azip!(
        density (densities),
        pos (positions),
        ref accel (accels),
    in {
        let cell = if let Some(cell) = serch.get_cell(&pos) { cell } else { return };
        let pressure_i = gas_constant * (density - rest_density);

        serch.for_each_neighbor(cell, 1, |p| {
            let pressure_j = gas_constant * (densities[p] - rest_density);
            let density_j = densities[p];
            let mass_j = masses[p];
            let two = cast::<f64, T>(2.0).unwrap();
            let r = pos - positions[p];
            *accel -= r * (mass_j * spiky.grad_w(pos.distance(positions[p])) * (pressure_j + pressure_i) / (two * density_j * density));
        });
    });
}

pub fn calculate_viscosity<T, N, NS>(p: &Processor, (kernel_size,viscosity, search): (T, T, &NS))
where
    T: Real + 'static,
    N: Dim<T>,
    <N as ArrayLength<T>>::ArrayType: Copy,
    NS: NeighborSearch<T, N>,
{
    let (densities, positions, velocities, accels, masses) = (
        p.read_property::<Density<T>>(),
        p.read_property::<Position<T, N>>(),
        p.read_property::<Velocity<T, N>>(),
        p.write_property::<Acceleration<T, N>>(),
        p.read_property::<Mass<T>>(),
    );

    let visc = kernel::Viscosity::new(kernel_size);

    par_azip!(
        ref accel (accels)
        density (densities),
        pos (positions),
        vel (velocities),
    in {
        let cell = if let Some(cell) = search.get_cell(&pos) { cell } else { return };

        // TODO: skip own?
        search.for_each_neighbor(cell, 1, |p| {
            let diff_vel = velocities[p] - vel;
            *accel += diff_vel * (viscosity * masses[p] / (density * densities[p]) * visc.laplace_w(pos.distance(positions[p])));
        });

    });
}

pub fn integrate_explicit_euler<T, N>(p: &Processor, timestep: T)
where
    T: Real + 'static,
    N: Dim<T>,
    <N as ArrayLength<T>>::ArrayType: Copy,
{
    let (pos, vel, accel) = (
        p.write_property::<Position<T, N>>(),
        p.write_property::<Velocity<T, N>>(),
        p.read_property::<Acceleration<T, N>>(),
    );

    par_azip!(
        mut pos (pos),
        mut vel (vel),
        accel (accel)
    in {
        *vel += accel * timestep;
        *pos += *vel * timestep;
    });
}
