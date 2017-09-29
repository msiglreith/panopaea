
//! Position-based dynamics

pub mod constraint;
pub mod property;

use self::property::*;

use cgmath::{InnerSpace, MetricSpace};
use sph::grid::BoundedGrid;
use sph::property::*;
use sph::kernel::{self, Kernel};
use particle::{Particles, Processor};
use typenum::U2;
use math::{Real, Dim, VectorN};
use num::Zero;
use ndarray::Zip;
use ndarray_parallel::prelude::*;

/// Initialize required properties for running position based dynamics simulations.
pub fn init<T, N>(particles: &mut Particles)
where
    T: Real + 'static,
    N: Dim<T>,
{
    particles.add_property::<Position<T, N>>();
    particles.add_property::<PredPosition<T, N>>();
    particles.add_property::<DeltaPos<T, N>>();
    particles.add_property::<Velocity<T, N>>();
    particles.add_property::<Acceleration<T, N>>();
    particles.add_property::<Lambda<T>>();
    particles.add_property::<Mass<T>>();
}

// Alg. 1 `Simulation Loop`, 2+3
/// Apply external forces and predict positions.
pub fn integrate<T, N>(p: &Processor, timestep: T)
where
    T: Real + 'static,
    N: Dim<T>,
{
    let (pred_positions, positions, velocities, accelerations) = (
        p.write_property::<PredPosition<T, N>>(),
        p.read_property::<Position<T, N>>(),
        p.write_property::<Velocity<T, N>>(),
        p.read_property::<Acceleration<T, N>>());

    Zip::from(pred_positions)
        .and(positions)
        .and(velocities)
        .and(accelerations)
        .par_apply(|pred_pos, pos, vel, accel| {
            *vel += accel * timestep;
            *pred_pos = pos + vel * timestep;
        });
}

// Alg. 1 `Simulation Loop`, 10
pub fn calculate_lambda<T>(p: &Processor, (rest_density, kernel_size, relaxation, grid): (T, T, T, &BoundedGrid<T, U2>))
where
    T: Real + 'static,
{
    let (lambdas, positions, masses) = (
        p.write_property::<Lambda<T>>(),
        p.read_property::<PredPosition<T, U2>>(),
        p.read_property::<Mass<T>>());

    let poly_6 = kernel::Poly6::new(kernel_size);
    let spiky = kernel::Spiky::new(kernel_size);

    Zip::indexed(lambdas)
        .and(masses)
        .and(positions)
        .par_apply(|i, lambda, &mass, &pos| {
            // Calculate density (Eq. 2)
            let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
            let mut density = mass * poly_6.w(T::zero());
            grid.for_each_neighbor(cell, 1, |p| {
                if p == i { return }
                density += masses[p] * poly_6.w(pos.distance(positions[p]));
            });

            // Fluid constraint (Eq. 1)
            let constraint = density/rest_density - T::one();
            if constraint <= T::zero() { // TODO: eps
                *lambda = T::zero();
                return;
            }

            let mut sum_grad = T::zero();
            let mut grad_i = VectorN::<T, U2>::zero();

            // Eq. 8
            grid.for_each_neighbor(cell, 1, |p| {
                if p == i { return }
                let diff = pos - positions[p];
                let grad_j = diff * (-masses[p] / rest_density * spiky.grad_w(pos.distance(positions[p])));
                grad_i -= grad_j;
                sum_grad += grad_j.magnitude2();
            });

            sum_grad += grad_i.magnitude2();

            *lambda = -constraint / (sum_grad + relaxation);
        });
}

// Alg. 1 `Simulation Loop`, 13
pub fn calculate_pos_delta<T>(p: &Processor, (rest_density, kernel_size, grid): (T, T, &BoundedGrid<T, U2>))
    where T: Real + 'static,
{
    let (lambdas, positions, delta_pos, masses) = (
        p.read_property::<Lambda<T>>(),
        p.read_property::<PredPosition<T, U2>>(),
        p.write_property::<DeltaPos<T, U2>>(),
        p.read_property::<Mass<T>>());

    let spiky = kernel::Spiky::new(kernel_size);

    par_azip!(
        index i,
        lambda_i (lambdas),
        pos (positions),
        mut delta_pos (delta_pos)
    in {
        // Calculate delta_p
        let cell = if let Some(cell) = grid.get_cell(&pos) { cell } else { return };
        *delta_pos = VectorN::<T, U2>::zero();
        grid.for_each_neighbor(cell, 1, |j| {
            if j == i { return }
            let diff = pos - positions[j];
            *delta_pos += diff * (masses[j] * (lambda_i + lambdas[j]) * spiky.grad_w(pos.distance(positions[j])));
        });
        *delta_pos /= rest_density;
    });
}

pub fn apply_delta<T, N>(p: &Processor)
where
    T: Real + 'static,
    N: Dim<T>,
{
    let (pos, delta) = (
        p.write_property::<PredPosition<T, N>>(),
        p.read_property::<DeltaPos<T, N>>());

    Zip::from(pos).and(delta).apply(|pos, delta| {
        *pos += delta;
    });
}

// Alg. 1 `Simulation Loop`, 21
pub fn update_velocity<T, N>(p: &Processor, timestep: T)
where
    T: Real + 'static,
    N: Dim<T>,
{
    let (velocities, positions, pred_positions) = (
        p.write_property::<Velocity<T, N>>(),
        p.read_property::<Position<T, N>>(),
        p.read_property::<PredPosition<T, N>>());

    Zip::from(velocities)
        .and(positions)
        .and(pred_positions)
        .par_apply(|vel, pos, pred_pos| {
            *vel = (pred_pos - pos) / timestep;
        });
}

// TODO: combine with other functions?
// Alg. 1 `Simulation Loop`, 23
pub fn update_position<T, N>(p: &Processor)
where
    T: Real + 'static,
    N: Dim<T>,
{
    let (positions, pred_positions) = (
        p.write_property::<Position<T, N>>(),
        p.read_property::<PredPosition<T, N>>());

    Zip::from(positions).and(pred_positions).par_apply(|pos, pred_pos| {
        *pos = pred_pos.clone();
    });
}

