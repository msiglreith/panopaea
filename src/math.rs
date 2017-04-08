
use cgmath::{self, MetricSpace};
use na::{self, DimName};
use generic_array::{ArrayLength, GenericArray};
use std;

use cgmath::BaseFloat;

pub fn vec2<N: na::Scalar>(x: N, y: N) -> na::Vector2<N> {
    na::Vector2::new(x, y)
}

pub fn linear<S: Real>(a0: S, a1: S, s: S) -> S {
    a0 * (S::one() - s) + a1 * s
}

pub fn bilinear<S: Real>(a00: S, a01: S, a10: S, a11: S, s: S, t: S) -> S {
    linear(linear(a00, a01, s), linear(a10, a11, s), t)
}

pub fn trilinear<S: Real>(
    a000: S, a001: S, a010: S, a011: S,
    a100: S, a101: S, a110: S, a111: S,
    s: S, t: S, u: S
) -> S {
    linear(bilinear(a000, a001, a010, a011, s, t), bilinear(a100, a101, a110, a111, s, t), u)
}

pub trait Real: BaseFloat + Send + Sync { }
impl<T> Real for T where T: BaseFloat + Send + Sync { }

pub trait Dim<S> : ArrayLength<S> + Clone + 'static { }
impl<T, S> Dim<S> for T where T: ArrayLength<S> + Clone + 'static { }

#[derive(Debug)]
pub struct VectorN<S: Real, N: Dim<S>>(pub GenericArray<S, N>);

unsafe impl<S: Real, N: Dim<S>> Send for VectorN<S, N> { }
unsafe impl<S: Real, N: Dim<S>> Sync for VectorN<S, N> { }

impl<S: Real, N: Dim<S>> Clone for VectorN<S, N> {
    fn clone(&self) -> Self {
        VectorN(self.0.clone())
    }
}

impl <S: Real, N: Dim<S>> VectorN<S, N> {
    pub fn from_element(elem: S) -> Self {
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

