
//! Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)

use alga::general::Real;
use particle::{self, Particles};

use super::kernel;

pub fn init_3d<T>(particles: &mut Particles)
    where T: Real + Clone + Default + 'static
{
    particles.add_property::<particle::property::Position3d<T>>();
    particles.add_property::<particle::property::Velocity3d<T>>();
    particles.add_property::<particle::property::Density<T>>();
}
