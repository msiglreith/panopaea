
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
use cgmath::{InnerSpace, MetricSpace};
use num::Zero;

pub fn init<T, N>(particles: &mut Particles)
    where T: Real + 'static,
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

// Alg. 1 `Simulation Loop`, 2
pub fn apply_forces<T>(p: &Processor, timestep: T)
    where T: Real + 'static,
{
    // println!("-- apply forces");
    let (velocities, accelerations) = (
        p.write_property::<Velocity<T, U2>>(),
        p.read_property::<Acceleration<T, U2>>());

    azip_par!(
        mut vel (velocities),
        accel (accelerations)
     in { *vel += accel * timestep; });

    // println!("velocity: {:?}", p.write_property::<Velocity<T, U2>>());
}

// Alg. 1 `Simulation Loop`, 3
// TODO: merge with `apply_forces`
pub fn predict_position<T>(p: &Processor, timestep: T)
    where T: Real,
{
    // println!("-- predict positions");
    let (pred_positions, positions, velocities) = (
        p.write_property::<PredPosition<T, U2>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<Velocity<T, U2>>());

    azip_par!(
        mut pred_pos (pred_positions),
        pos (positions),
        vel (velocities)
     in { *pred_pos = pos + vel * timestep; });

    // println!("pred_pose: {:?}", p.write_property::<PredPosition<T, U2>>());
}

// Alg. 1 `Simulation Loop`, 10
pub fn calculate_lambda<T>(p: &Processor, (rest_density, kernel_size, relaxation, grid): (T, T, T, &BoundedGrid<T, U2>))
    where T: Real + 'static,
{
    let (lambdas, positions, masses) = (
        p.write_property::<Lambda<T>>(),
        p.read_property::<PredPosition<T, U2>>(),
        p.read_property::<Mass<T>>());

    let poly_6 = kernel::Poly6::new(kernel_size);
    let spiky = kernel::Spiky::new(kernel_size);

    azip_indexed_par!(
        mut lambda (lambdas),
        mass (masses),
        pos (positions),
    index i in {
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

    azip_indexed_par!(
        lambda_i (lambdas),
        pos (positions),
        mut delta_pos (delta_pos)
    index i in {
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

pub fn apply_delta<T>(p: &Processor)
    where T: Real + 'static,
{
    let (pos, delta) = (
        p.write_property::<PredPosition<T, U2>>(),
        p.read_property::<DeltaPos<T, U2>>());

    azip_par!(
        mut pos (pos),
        delta (delta)
    in { *pos += delta; });
}

// Alg. 1 `Simulation Loop`, 21
pub fn update_velocity<T>(p: &Processor, timestep: T)
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
pub fn update_position<T>(p: &Processor)
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

