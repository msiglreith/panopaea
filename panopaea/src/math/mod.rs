
use cgmath::BaseFloat;
use na;
use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Dimension, Ix1};
use num;
use generic_array::ArrayLength;
use rand;

pub mod interp;
pub mod vector_n;
pub mod wavelet;

pub use self::interp::{linear, bilinear, trilinear};
pub use self::vector_n::VectorN;

pub fn vec2<N: na::Scalar>(x: N, y: N) -> na::Vector2<N> {
    na::Vector2::new(x, y)
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

impl<A, D: Dimension> LinearView for Array<A, D> {
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

pub trait Real: BaseFloat + rand::Rand + 'static + Send + Sync {
    fn new<U: num::NumCast>(other: U) -> Self {
        num::NumCast::from(other).unwrap()
    }
}

impl<T> Real for T where T: BaseFloat + rand::Rand + 'static + Send + Sync { }

pub trait MulOut {
    type RHS;
    type Output;

    fn mul_out(&self, y: &mut Self::Output, x: &Self::RHS);
}
