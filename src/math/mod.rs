
use cgmath::BaseFloat;
use na;
use ndarray::{ArrayBase, ArrayView, ArrayViewMut, Dimension, Ix1};
use generic_array::ArrayLength;

pub mod vector_n;

pub use self::vector_n::VectorN;

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

pub trait LinearView {
    type Elem;
    fn view_linear(&self) -> ArrayView<Self::Elem, Ix1>;
    fn view_linear_mut(&mut self) -> ArrayViewMut<Self::Elem, Ix1>;
}

pub trait LinearViewReal<A: Real> : LinearView<Elem = A> {
    fn dot_linear<Rhs: LinearView<Elem = A>>(&self, rhs: &Rhs) -> A {
        self.view_linear().dot(&rhs.view_linear())
    }

    fn norm_max(&self) -> A {
        let mut max = A::zero();
        for &x in self.view_linear() {
            max = if x > max { x } else if -x > max { -x } else { max };
        }
        max
    }
}

impl<T, A: Real> LinearViewReal<A> for T where T: LinearView<Elem = A> { }

impl<A, D: Dimension> LinearView for ArrayBase<Vec<A>, D> {
    type Elem = A;
    fn view_linear(&self) -> ArrayView<A, Ix1> {
        unsafe { ArrayView::<A, Ix1>::from_shape_ptr(self.len(), self.as_ptr()) }
    }

    fn view_linear_mut(&mut self) -> ArrayViewMut<A, Ix1> {
        unsafe { ArrayViewMut::<A, Ix1>::from_shape_ptr(self.len(), self.as_mut_ptr()) }
    }
}

pub trait Dim<S> : ArrayLength<S> + Clone + 'static { }
impl<T, S> Dim<S> for T where T: ArrayLength<S> + Clone + 'static { }

pub trait Real: BaseFloat + 'static + Send + Sync { }
impl<T> Real for T where T: BaseFloat + 'static + Send + Sync { }

pub trait MulOut {
    type RHS;
    type Output;

    fn mul_out(&self, y: &mut Self::Output, x: &Self::RHS);
}
