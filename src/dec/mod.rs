//! Discrete exterior calculus (DEC)

use std::ops::{Deref, DerefMut};

pub mod grid;
pub mod manifold;

pub struct Primal<T>(T);

impl<T> Deref for Primal<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Primal<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

pub struct Dual<T>(T);

impl<T> Deref for Dual<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Dual<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
