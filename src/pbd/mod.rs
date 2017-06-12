
//! Position-based dynamics

pub mod constraint;
pub mod property;

use self::property::*;

use sph::grid::BoundedGrid;
use sph::property::*;
use particle::{Particles, Processor};
use typenum::U2;
use math::{Real, Dim};
use rayon::prelude::*;

pub fn init<T, N>(particles: &mut Particles)
    where T: Real + 'static,
          N: Dim<T>,
{
    particles.add_property::<Position<T, N>>();
    particles.add_property::<PredPosition<T, N>>();
    particles.add_property::<Velocity<T, N>>();
    particles.add_property::<Acceleration<T, N>>();
    particles.add_property::<Density<T>>();
    particles.add_property::<Mass<T>>();
}

pub fn predict_position<T>(timestep: T, p: Processor)
    where T: Real + 'static,
{
    let (mut pred_positions, positions, velocities) = (
        p.write_property::<PredPosition<T, U2>>(),
        p.read_property::<Position<T, U2>>(),
        p.read_property::<Velocity<T, U2>>(),
    );

    azip_par!(
        mut pred_pos (pred_positions),
        pos (positions),
        vel (velocities)
     in { *pred_pos += pos + vel * timestep; });
}