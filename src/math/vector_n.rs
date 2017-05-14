
use cgmath::{Vector2, MetricSpace};
use generic_array::GenericArray;
use generic_array::typenum::{U2, U3};
use std::ops::{Deref, DerefMut};

use super::{Dim, Real};

#[derive(Debug)]
pub struct VectorN<S, N: Dim<S>>(pub GenericArray<S, N>);

unsafe impl<S: Send, N: Dim<S>> Send for VectorN<S, N> { }
unsafe impl<S: Sync, N: Dim<S>> Sync for VectorN<S, N> { }

impl<S: Clone, N: Dim<S>> Clone for VectorN<S, N> {
    fn clone(&self) -> Self {
        VectorN(self.0.clone())
    }
}

impl <S: Clone, N: Dim<S>> VectorN<S, N> {
    pub fn from_elem(elem: S) -> Self {
        let len = N::to_usize();
        VectorN(GenericArray::clone_from_slice(&vec![elem; len]))
    }
}

impl<S: Real, N: Dim<S>> MetricSpace for VectorN<S, N> {
    type Metric = S;

    fn distance2(self, other: Self) -> Self::Metric {
        unimplemented!()
    }

    fn distance(self, other: Self) -> Self::Metric {
        unimplemented!()
    }
}

impl<S, N: Dim<S>> Deref for VectorN<S, N> {
    type Target = [S];
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<S, N: Dim<S>> DerefMut for VectorN<S, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

impl<S: Copy> Into<[S; 2]> for VectorN<S, U2> {
    fn into(self) -> [S; 2] {
        [self[0], self[1]]
    }
}

pub fn vec2<S: Clone>(x: S, y: S) -> VectorN<S, U2> {
    VectorN(GenericArray::clone_from_slice(&[x, y]))
}
