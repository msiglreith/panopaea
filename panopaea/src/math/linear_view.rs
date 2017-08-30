
use ndarray::{Array, ArrayView, ArrayViewMut, Dimension, Ix1};
use super::Real;

pub trait LinearView {
    type Elem;
    fn view_linear(&self) -> ArrayView<Self::Elem, Ix1>;
    fn view_linear_mut(&mut self) -> ArrayViewMut<Self::Elem, Ix1>;
}

pub trait LinearViewReal<A: Real>: LinearView<Elem = A> {
    fn dot_linear<Rhs: LinearView<Elem = A>>(&self, rhs: &Rhs) -> A {
        self.view_linear().dot(&rhs.view_linear())
    }

    fn norm_max(&self) -> A {
        let mut max = A::zero();
        for &x in self.view_linear() {
            max = if x > max {
                x
            } else if -x > max {
                -x
            } else {
                max
            };
        }
        max
    }
}

impl<T, A: Real> LinearViewReal<A> for T
where
    T: LinearView<Elem = A>,
{
}

impl<A, D: Dimension> LinearView for Array<A, D> {
    type Elem = A;
    fn view_linear(&self) -> ArrayView<A, Ix1> {
        unsafe { ArrayView::<A, Ix1>::from_shape_ptr(self.len(), self.as_ptr()) }
    }

    fn view_linear_mut(&mut self) -> ArrayViewMut<A, Ix1> {
        unsafe { ArrayViewMut::<A, Ix1>::from_shape_ptr(self.len(), self.as_mut_ptr()) }
    }
}
