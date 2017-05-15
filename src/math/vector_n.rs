
use cgmath::{BaseNum, Vector2, MetricSpace, VectorSpace};
use generic_array::GenericArray;
use generic_array::typenum::{U2, U3};
use std::ops::{Add, AddAssign, Deref, DerefMut, Sub, SubAssign, Mul, Div, Rem};
use num::Zero;

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

impl <S: Copy, N: Dim<S>> Copy for VectorN<S, N> where GenericArray<S, N>: Copy { }

impl <S: Clone + Zero, N: Dim<S>> Zero for VectorN<S, N> {
    #[inline]
    fn zero() -> VectorN<S, N> {
        Self::from_elem(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        unimplemented!()
        // *self == Self::zero()
    }
}

impl <S, N: Dim<S>> Add for VectorN<S, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl <S: AddAssign + Copy, N: Dim<S>> AddAssign for VectorN<S, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N::to_usize() {
            self[i] += rhs[i];
        }
    }
}

impl <S, N: Dim<S>> Sub for VectorN<S, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl <S: SubAssign + Copy, N: Dim<S>> SubAssign for VectorN<S, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N::to_usize() {
            self[i] -= rhs[i];
        }
    }
}

impl <S, N: Dim<S>> Mul<S> for VectorN<S, N> {
    type Output = Self;
    fn mul(self, rhs: S) -> Self::Output {
        unimplemented!()
    }
}

impl <S, N: Dim<S>> Div<S> for VectorN<S, N> {
    type Output = Self;
    fn div(self, rhs: S) -> Self::Output {
        unimplemented!()
    }
}

impl <S, N: Dim<S>> Rem<S> for VectorN<S, N> {
    type Output = Self;
    fn rem(self, rhs: S) -> Self::Output {
        unimplemented!()
    }
}

impl<S: BaseNum, N: Dim<S>> VectorSpace for VectorN<S, N>
    where VectorN<S, N>: Copy {
    type Scalar = S;
}

impl<'a, S: Real, N: Dim<S>> MetricSpace for &'a VectorN<S, N> {
    type Metric = S;

    fn distance2(self, other: Self) -> Self::Metric {
        let mut dist = S::zero();
        for i in 0..N::to_usize() {
            dist += (self[i]-other[i]).powi(2)
        }
        dist 

        /*
        self.0.iter()
            .zip(other.0.iter())
            .map(|(&x, &y)| (x-y).powi(2))
            .fold(S::zero(), |a, b| a + b)
        */
    }

    fn distance(self, other: Self) -> Self::Metric {
        // self.distance(other).sqrt()
        let mut dist = S::zero();
        for i in 0..N::to_usize() {
            dist += (self[i]-other[i]).powi(2)
        }
        dist.sqrt()
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
