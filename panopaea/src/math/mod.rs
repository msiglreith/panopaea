
use cgmath::BaseFloat;
use na;
use num;
use generic_array::ArrayLength;
use rand;

pub mod integration;
pub mod interp;
pub mod linear_view;
pub mod vector_n;
pub mod wavelet;

pub use self::interp::{bilinear, linear, trilinear};
pub use self::linear_view::{LinearView, LinearViewReal};
pub use self::vector_n::VectorN;

pub fn vec2<N: na::Scalar>(x: N, y: N) -> na::Vector2<N> {
    na::Vector2::new(x, y)
}

pub trait Dim<S>: ArrayLength<S> + ArrayLength<usize> + Clone + Copy + 'static {}
impl<T, S> Dim<S> for T
where
    T: ArrayLength<S> + ArrayLength<usize> + Clone + Copy + 'static,
{
}

pub trait Real: BaseFloat + rand::Rand + 'static + Send + Sync {
    fn new<U: num::NumCast>(other: U) -> Self {
        num::NumCast::from(other).unwrap()
    }
}

impl<T> Real for T
where
    T: BaseFloat + rand::Rand + 'static + Send + Sync,
{
}

pub trait MulOut {
    type RHS;
    type Output;

    fn mul_out(&self, y: &mut Self::Output, x: &Self::RHS);
}
