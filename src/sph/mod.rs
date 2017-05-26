
//! Smoothed Particle hydrodynamics
//!
//! References:
//!     [MDM03] Matthias Müller, David Charypar, and Markus Gross, 2003,
//!             Particle-based fluid simulation for interactive applications,
//!             In Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation (SCA '03),
//!             Eurographics Association, Aire-la-Ville, Switzerland, Switzerland, 154-159

pub mod grid;
pub mod kernel;
pub mod wcsph;

use math::{Real, Dim};
use particle::{Particles, Property};
use rayon::prelude::*;

pub mod property {
    //! Common particle properties
    use math::{Real, Dim, VectorN};
    use particle::Property;

    pub struct Position<T: Real, N: Dim<T>>(pub VectorN<T, N>);
    impl<T: Real, N: Dim<T>> Property for Position<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    pub struct Velocity<T: Real, N: Dim<T>>(pub VectorN<T, N>);
    impl<T: Real, N: Dim<T>> Property for Velocity<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    pub struct Acceleration<T: Real, N: Dim<T>>(pub VectorN<T, N>);
    impl<T: Real, N: Dim<T>> Property for Acceleration<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    pub struct Mass<T: Real>(pub T);
    impl<T: Real> Property for Mass<T> {
        type Subtype = T;
        fn new() -> Self::Subtype {
            T::zero()
        }
    }

    pub struct Density<T: Real>(pub T);
    impl<T: Real> Property for Density<T> {
        type Subtype = T;
        fn new() -> Self::Subtype {
            T::zero()
        }
    }

    pub struct Pressure<T: Real>(pub T);
    impl<T: Real> Property for Pressure<T> {
        type Subtype = T;
        fn new() -> Self::Subtype {
            T::zero()
        }
    }
}

// TODO: move this into Particles to allow reseting all kind of properties
pub fn reset_acceleration<T, N>(particles: &mut Particles)
    where T: Real, N: Dim<T>
{
    use self::property::Acceleration;
    particles.run(|p| {
        let mut accel = p.write_property::<Acceleration<T, N>>();
        accel.par_iter_mut().for_each(|mut a| *a = Acceleration::<T, N>::new() );
    });
}
