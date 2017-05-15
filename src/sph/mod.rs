
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

use math::{Real, Dim, VectorN};
use particle::{Particles, Property};
use rayon::prelude::*;

pub mod property {
    //! Common particle properties
    use std::ops::{Deref, DerefMut};
    use math::{Real, Dim, VectorN};
    use generic_array::{ArrayLength};
    use particle::Property;

    #[derive(Clone, Debug)]
    pub struct Position<T: Real, N: Dim<T>>(pub VectorN<T, N>);

    impl<T: Real, N: Dim<T>> Property for Position<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    impl<T: Real, N: Dim<T>> Copy for Position<T, N>
        where VectorN<T, N>: Copy { }

    impl<T, N> Deref for Position<T, N>
        where T: Real, N: Dim<T>
    {
        type Target = VectorN<T, N>;
        fn deref(&self) -> &VectorN<T, N> {
            &self.0
        }
    }

    impl<T, N> DerefMut for Position<T, N>
        where T: Real, N: Dim<T>
    {
        fn deref_mut(&mut self) -> &mut VectorN<T, N> {
            &mut self.0
        }
    }

    impl<T, N> Default for Position<T, N>
        where T: Real, N: Dim<T>
    {
        fn default() -> Self {
            Position(VectorN::from_elem(T::zero()))
        }
    }

    #[derive(Clone, Debug)]
    pub struct Velocity<T: Real, N: Dim<T>>(pub VectorN<T, N>);

    impl<T: Real, N: Dim<T>> Property for Velocity<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    impl<T: Real, N: Dim<T>> Copy for Velocity<T, N>
        where VectorN<T, N>: Copy { }

    impl<T, N> Deref for Velocity<T, N>
        where T: Real, N: Dim<T>
    {
        type Target = VectorN<T, N>;
        fn deref(&self) -> &VectorN<T, N> {
            &self.0
        }
    }

    impl<T, N> DerefMut for Velocity<T, N>
        where T: Real, N: Dim<T>
    {
        fn deref_mut(&mut self) -> &mut VectorN<T, N> {
            &mut self.0
        }
    }

    impl<T, N> Default for Velocity<T, N>
        where T: Real, N: Dim<T>
    {
        fn default() -> Self {
            Velocity(VectorN::from_elem(T::zero()))
        }
    }

    #[derive(Clone, Debug)]
    pub struct Acceleration<T: Real, N: Dim<T>>(pub VectorN<T, N>);

    impl<T: Real, N: Dim<T>> Property for Acceleration<T, N> {
        type Subtype = VectorN<T, N>;
        fn new() -> Self::Subtype {
            VectorN::from_elem(T::zero())
        }
    }

    impl<T, N> Deref for Acceleration<T, N>
        where T: Real, N: Dim<T>
    {
        type Target = VectorN<T, N>;
        fn deref(&self) -> &VectorN<T, N> {
            &self.0
        }
    }

    impl<T, N> DerefMut for Acceleration<T, N>
        where T: Real, N: Dim<T>
    {
        fn deref_mut(&mut self) -> &mut VectorN<T, N> {
            &mut self.0
        }
    }

    impl<T, N> Default for Acceleration<T, N>
        where T: Real, N: Dim<T>
    {
        fn default() -> Self {
            Acceleration(VectorN::from_elem(T::zero()))
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct Mass<T: Real>(pub T);

    impl<T: Real> Property for Mass<T> {
        type Subtype = T;
        fn new() -> Self::Subtype {
            T::zero()
        }
    }

    #[derive(Copy, Clone, Debug)]
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

pub fn reset_acceleration<T, N>(particles: &mut Particles)
    where T: Real, N: Dim<T>
{
    use self::property::Acceleration;
    particles.run(|p| {
        let mut accel = p.write_property::<Acceleration<T, N>>().unwrap();
        accel.par_iter_mut().for_each(|mut a| *a = Acceleration::<T, N>::new() );
    });
}
