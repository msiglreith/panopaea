
//! Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)

use math::{Dim, Real};
use generic_array::ArrayLength;
use particle::{self, Particles};
use rayon::prelude::*;

use super::kernel;
use super::property::*;

pub fn init<T, N, S>(particles: &mut Particles)
    where T: Real + 'static,
          N: Dim<T>,
{
    particles.add_property::<Position<T, N>>();
    particles.add_property::<Velocity<T, N>>();
    particles.add_property::<Density<T>>();
    particles.add_property::<Mass<T>>();
}

pub fn compute_density<T, N, S>(particles: &mut Particles)
    where T: Real + 'static,
          N: Dim<T>,
{
    particles.run(|p| {
        let (mut density, position, mass) = (
            p.write_property::<Density<T>>().unwrap(),
            p.read_property::<Position<T, N>>().unwrap(),
            p.read_property::<Mass<T>>().unwrap(),
        );

        density.par_iter_mut()
           .zip(position.par_iter())
           .zip(mass.par_iter())
           .for_each(|((mut density, position), &mass)| {
                // TODO
                *density = Density(*mass);
            });
    })
}
