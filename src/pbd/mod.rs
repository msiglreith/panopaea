
//! Position-based dynamics

pub mod constraint;
pub mod property;

use self::property::*;

use sph::grid::BoundedGrid;
use sph::property::*;
use sph::kernel::{self, Kernel};
use particle::{Particles, Processor};
use typenum::U2;
use math::{Real, Dim, VectorN};
use rayon::prelude::*;
use cgmath::MetricSpace;
use num::Zero;

pub fn init<T, N>(particles: &mut Particles)
    where T: Real + 'static,
          N: Dim<T>,
{
    particles.add_property::<Position<T, N>>();
    particles.add_property::<PredPosition<T, N>>();
    particles.add_property::<Velocity<T, N>>();
    particles.add_property::<Acceleration<T, N>>();
    particles.add_property::<Lambda<T>>();
    particles.add_property::<Mass<T>>();
}

// Alg. 1 `Simulation Loop`, 2
pub fn apply_forces<T>(timestep: T, p: Processor)
    where T: Real + 'static,
{
    let (velocities, accelerations) = (
        p.write_property::<Velocity<T, U2>>(),
        p.read_property::<Acceleration<T, U2>>());

    azip_par!(
        mut vel (velocities),
        accel (accelerations)
     in { *vel += accel* timestep; });
}

// Alg. 1 `Simulation Loop`, 10
pub fn calculate_lambda<T>(rest_density: T, kernel_size: T, relaxation: T, grid: &BoundedGrid<T, U2>, p: Processor)
    where T: Real + 'static,
{
    let (mut lambdas, positions, masses) = (
        p.write_property::<Lambda<T>>(),
        p.read_property::<PredPosition<T, U2>>(),
        p.read_property::<Mass<T>>());

    let poly_6 = kernel::Poly6::new(kernel_size);
    let spiky = kernel::Spiky::new(kernel_size);

    lambdas.par_iter_mut().enumerate()
        .zip(masses.par_iter())
        .zip(positions.par_iter())
        .for_each(|(((i, mut lambda), &mass), pos)| {
            // TODO: do we need density for each?

            // Calculate density (Eq. 2)
            let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
            let mut density = mass * poly_6.w(T::zero());
            grid.for_each_neighbor(cell, 1, |p| {
                if p == i { return }
                density += masses[p] * poly_6.w(pos.distance(&positions[p]));
            });

            // Fluid constraint (Eq. 1)
            let constraint = density/rest_density - T::one();
            if constraint == T::zero() { // TODO: eps
                *lambda = T::zero();
                return;
            }

            let mut sum_grad = T::zero();
            let mut grad_i = VectorN::<T, U2>::zero();

            // Eq. 8
            grid.for_each_neighbor(cell, 1, |p| {
                let diff = *pos - positions[p];
                let grad_j = diff * (-masses[p] / rest_density * spiky.grad_w(pos.distance(&positions[p])));
                grad_i -= grad_j;
                // TODO: sum grad_j norm
            });

            // TOOD: sum grad_i norm

            *lambda = -constraint / (sum_grad + relaxation); 
         });
}

// Alg. 1 `Simulation Loop`, 3
pub fn predict_position<T>(timestep: T, p: Processor)
    where T: Real + 'static,
{
    let (pred_positions, positions, velocities) = (
        p.write_property::<PredPosition<T, U2>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<Velocity<T, U2>>());

    azip_par!(
        mut pred_pos (pred_positions),
        pos (positions),
        vel (velocities)
     in { *pred_pos += pos + vel * timestep; });
}

// Alg. 1 `Simulation Loop`, 21
pub fn update_velocity<T>(timestep: T, p: Processor)
    where T: Real + 'static
{
    let (velocities, positions, pred_positions) = (
        p.write_property::<Velocity<T, U2>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<PredPosition<T, U2>>());

    azip_par!(
        mut vel (velocities),
        pos (positions),
        pred_pos (pred_positions)
    in { *vel = (pred_pos - pos) / timestep; });
}

// TODO: combine with other functions?
// Alg. 1 `Simulation Loop`, 23
pub fn update_position<T>(p: Processor)
    where T: Real + 'static
{
    let (positions, pred_positions) = (
        p.write_property::<Position<T, U2>>(),
        p.read_property::<PredPosition<T, U2>>());

    azip_par!(
        mut pos (positions),
        pred_pos (pred_positions)
    in { *pos = pred_pos; });
}

