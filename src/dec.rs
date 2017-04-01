//! Discrete exterior calculus (DEC)

use ndarray::{Array, Ix2, LinalgScalar, Zip};
use sparse;

use std::ops::{Deref, DerefMut, Neg};

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


pub trait Manifold2d<T> {
    type Form0;
    type Form1;
    type Form2;

    fn derivative_0(&self, &mut Self::Form1, &Self::Form0);
    fn derivative_1(&self, &mut Self::Form2, &Self::Form1);

    fn hodge_1_primal(&self, dual: &mut Self::Form1, primal: &Self::Form1);
    fn hodge_1_dual(&self, primal: &mut Self::Form1, dual: &Self::Form1);
}

pub struct Grid;

impl<T> Manifold2d<T> for Grid
    where T: LinalgScalar + Neg<Output = T>
{
    type Form0 = Array<T, Ix2>;
    type Form1 = (Array<T, Ix2>, Array<T, Ix2>);
    type Form2 = Array<T, Ix2>;

    fn derivative_0(&self, edges: &mut Self::Form1, vertices: &Self::Form0) {
        Zip::from(&mut edges.0)
            .and(vertices.slice(s![.., ..-1]))
            .and(vertices.slice(s![.., 1..]))
            .apply(|edge, &v0, &v1| {
                *edge = v1 - v0;
            });

        Zip::from(&mut edges.1)
            .and(vertices.slice(s![..-1, ..]))
            .and(vertices.slice(s![1.., ..]))
            .apply(|edge, &v0, &v1| {
                *edge = v1 - v0;
            });
    }

    fn derivative_1(&self, faces: &mut Self::Form2, edges: &Self::Form1) {
        Zip::from(faces)
            .and(edges.0.slice(s![..-1, ..]))
            .and(edges.0.slice(s![1.., ..]))
            .and(edges.1.slice(s![.., ..-1]))
            .and(edges.1.slice(s![.., 1..]))
            .apply(|face, &top, &bottom, &left, &right| {
                *face = bottom - top + left - right;
            });
    }

    fn hodge_1_primal(&self, dual: &mut Self::Form1, primal: &Self::Form1) {
        // boundary handling: top
        Zip::from(dual.0.slice_mut(s![..1, ..]))
            .and(primal.0.slice(s![..1, ..]))
            .apply(|dual, &primal| {
                let two = T::one() + T::one();
                *dual = primal / two;
            });

        // boundary handling: left
        Zip::from(dual.1.slice_mut(s![.., ..1]))
            .and(primal.1.slice(s![.., ..1]))
            .apply(|dual, &primal| {
                let two = T::one() + T::one();
                *dual = primal / two;
            });

        Zip::from(dual.0.slice_mut(s![1..-1, ..]))
            .and(primal.0.slice(s![1..-1, ..]))
            .apply(|dual, &primal| {
                *dual = primal;
            });

        Zip::from(dual.1.slice_mut(s![.., 1..-1]))
            .and(primal.1.slice(s![.., 1..-1]))
            .apply(|dual, &primal| {
                *dual = primal;
            });

        // boundary handling: bottom
        Zip::from(dual.0.slice_mut(s![-1.., ..]))
            .and(primal.0.slice(s![-1.., ..]))
            .apply(|dual, &primal| {
                let two = T::one() + T::one();
                *dual = primal / two;
            });

        // boundary handling: right
        Zip::from(dual.1.slice_mut(s![.., -1..]))
            .and(primal.1.slice(s![.., -1..]))
            .apply(|dual, &primal| {
                let two = T::one() + T::one();
                *dual = primal / two;
            });
    }

    fn hodge_1_dual(&self, primal: &mut Self::Form1, dual: &Self::Form1) {
        // boundary handling: top/left
        Zip::from(primal.0.slice_mut(s![..1, ..]))
            .and(dual.0.slice(s![..1, ..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = T::zero() - dual * two;
            });

        Zip::from(primal.1.slice_mut(s![.., ..1]))
            .and(dual.1.slice(s![.., ..1]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });

        Zip::from(primal.0.slice_mut(s![1..-1, ..]))
            .and(dual.0.slice(s![1..-1, ..]))
            .apply(|primal, &dual| {
                *primal = -dual;
            });

        Zip::from(primal.1.slice_mut(s![.., 1..-1]))
            .and(dual.1.slice(s![.., 1..-1]))
            .apply(|primal, &dual| {
                *primal = -dual;
            });

        // boundary handling: bottom/right
        Zip::from(primal.0.slice_mut(s![-1.., ..]))
            .and(dual.0.slice(s![-1.., ..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });

        Zip::from(primal.1.slice_mut(s![.., -1..]))
            .and(dual.1.slice(s![.., -1..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });
    }
}
