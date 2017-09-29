
use cgmath::{ApproxEq, BaseNum, InnerSpace, MetricSpace, VectorSpace};
use generic_array::GenericArray;
use generic_array::typenum::{U2};
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, Rem, Sub, SubAssign};
use std::mem;
use num::Zero;

use super::{Dim, Real};

#[derive(Debug)]
pub struct VectorN<S, N: Dim<S>>(pub GenericArray<S, N>);

unsafe impl<S: Send, N: Dim<S>> Send for VectorN<S, N> {}
unsafe impl<S: Sync, N: Dim<S>> Sync for VectorN<S, N> {}

impl<S: Clone, N: Dim<S>> Clone for VectorN<S, N> {
    fn clone(&self) -> Self {
        VectorN(self.0.clone())
    }
}

impl<S: Clone, N: Dim<S>> VectorN<S, N> {
    pub fn from_elem(elem: S) -> Self {
        let len = N::to_usize();
        VectorN(GenericArray::clone_from_slice(&vec![elem; len]))
    }
}

impl<S: Copy, N: Dim<S>> Copy for VectorN<S, N>
where
    GenericArray<S, N>: Copy,
{
}

impl<S: Copy + Zero, N: Dim<S>> Zero for VectorN<S, N> {
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

impl<S: Add<Output = S> + Copy, N: Dim<S>> Add for VectorN<S, N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N::to_usize() {
            self[i] = self[i] + rhs[i];
        }
        self
    }
}

impl<'a, S: Add<Output = S> + Copy, N: Dim<S>> Add for &'a VectorN<S, N> {
    type Output = VectorN<S, N>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result: VectorN<S, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            result[i] = self[i] + rhs[i];
        }
        result
    }
}

impl<'a, S: Add<Output = S> + Copy, N: Dim<S>> Add<VectorN<S, N>> for &'a VectorN<S, N> {
    type Output = VectorN<S, N>;
    fn add(self, rhs: VectorN<S, N>) -> Self::Output {
        let mut result: VectorN<S, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            result[i] = self[i] + rhs[i];
        }
        result
    }
}

impl<S: AddAssign + Copy, N: Dim<S>> AddAssign for VectorN<S, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N::to_usize() {
            self[i] += rhs[i];
        }
    }
}

impl<'a, S: AddAssign + Copy, N: Dim<S>> AddAssign<&'a VectorN<S, N>> for VectorN<S, N> {
    fn add_assign(&mut self, rhs: &'a VectorN<S, N>) {
        for i in 0..N::to_usize() {
            self[i] += rhs[i];
        }
    }
}

impl<S: Sub<Output = S> + Copy, N: Dim<S>> Sub for VectorN<S, N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N::to_usize() {
            self[i] = self[i] - rhs[i];
        }
        self
    }
}

impl<'a, S: Sub<Output = S> + Copy, N: Dim<S>> Sub for &'a VectorN<S, N> {
    type Output = VectorN<S, N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result: VectorN<S, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            result[i] = self[i] - rhs[i];
        }
        result
    }
}

impl<S: SubAssign + Copy, N: Dim<S>> SubAssign for VectorN<S, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N::to_usize() {
            self[i] -= rhs[i];
        }
    }
}

impl<S: Mul<Output = S> + Copy, N: Dim<S>> Mul<S> for VectorN<S, N> {
    type Output = Self;
    fn mul(mut self, rhs: S) -> Self::Output {
        for i in 0..N::to_usize() {
            self[i] = self[i] * rhs;
        }
        self
    }
}

impl<'a, S: Mul<Output = S> + Copy, N: Dim<S>> Mul<S> for &'a VectorN<S, N> {
    type Output = VectorN<S, N>;
    fn mul(self, rhs: S) -> Self::Output {
        let mut result: VectorN<S, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            result[i] = self[i] * rhs;
        }
        result
    }
}

impl<'a, S: Mul<Output = S> + Copy, N: Dim<S>> Mul<S> for &'a mut VectorN<S, N> {
    type Output = VectorN<S, N>;
    fn mul(mut self, rhs: S) -> Self::Output {
        let mut result: VectorN<S, N> = unsafe { mem::uninitialized() };
        for i in 0..N::to_usize() {
            result[i] = self[i] * rhs;
        }
        result
    }
}

impl<S: Div<Output = S> + Copy, N: Dim<S>> Div<S> for VectorN<S, N> {
    type Output = Self;
    fn div(mut self, rhs: S) -> Self::Output {
        for i in 0..N::to_usize() {
            self[i] = self[i] / rhs;
        }
        self
    }
}

impl<S: DivAssign + Copy, N: Dim<S>> DivAssign<S> for VectorN<S, N> {
    fn div_assign(&mut self, rhs: S) {
        for i in 0..N::to_usize() {
            self[i] /= rhs;
        }
    }
}

impl<S, N: Dim<S>> Rem<S> for VectorN<S, N> {
    type Output = Self;
    fn rem(self, _rhs: S) -> Self::Output {
        unimplemented!()
    }
}

impl<S: Real, N: Dim<S>> ApproxEq for VectorN<S, N> {
    type Epsilon = S::Epsilon;
    #[inline]
    fn default_epsilon() -> S::Epsilon {
        S::default_epsilon()
    }
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }
    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        let mut result = true;
        for i in 0..N::to_usize() {
            result = result && S::relative_eq(&self[i], &other[i], epsilon, max_relative);
        }
        result
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        let mut result = true;
        for i in 0..N::to_usize() {
            result = result && S::ulps_eq(&self[i], &other[i], epsilon, max_ulps);
        }
        result
    }
}

impl<S: BaseNum, N: Dim<S>> VectorSpace for VectorN<S, N>
where
    VectorN<S, N>: Copy,
{
    type Scalar = S;
}

impl<S: Real, N: Dim<S>> InnerSpace for VectorN<S, N>
where
    VectorN<S, N>: Copy,
{
    fn dot(self, other: Self) -> S {
        let mut dot = S::zero();
        for i in 0..N::to_usize() {
            dot += self[i] * other[i];
        }
        dot
    }
}

impl<S: Real, N: Dim<S>> MetricSpace for VectorN<S, N> {
    type Metric = S;

    fn distance2(self, other: Self) -> Self::Metric {
        let mut dist = S::zero();
        for i in 0..N::to_usize() {
            dist += (self[i] - other[i]).powi(2)
        }
        dist
    }

    fn distance(self, other: Self) -> Self::Metric {
        let mut dist = S::zero();
        for i in 0..N::to_usize() {
            dist += (self[i] - other[i]).powi(2)
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
