
use sparse::{SparseMatrix};
use std::marker::PhantomData;
use std::ops::Mul;

/// Storage type for 0-simplices (vertex).
/// A differential primal 0-form is stored for each 0-simplex.
/// Storage type for 1-simplices (edge).
/// A differential primal 1-form is stored for each 1-simplex.
/// Storage type for 2-simplices (face).
/// A differential primal 2-form is stored for each 2-simplex.

pub trait DecDomain2d<T> {
    type Simplex0;
    type Simplex1;
    type Simplex2;
}

pub trait Manifold2d<T>
    : DecDomain2d<T>
    + Hodge<T, <Self as DecDomain2d<T>>::Simplex0>
    + Hodge<T, <Self as DecDomain2d<T>>::Simplex1>
    + Hodge<T, <Self as DecDomain2d<T>>::Simplex2>
    + DerivativePrimal<T, <Self as DecDomain2d<T>>::Simplex0, <Self as DecDomain2d<T>>::Simplex1>
    + DerivativePrimal<T, <Self as DecDomain2d<T>>::Simplex1, <Self as DecDomain2d<T>>::Simplex2>
    + DerivativeDual<T, <Self as DecDomain2d<T>>::Simplex2, <Self as DecDomain2d<T>>::Simplex1>
    + DerivativeDual<T, <Self as DecDomain2d<T>>::Simplex1, <Self as DecDomain2d<T>>::Simplex0>
{
    ///
    fn num_elem_0(&self) -> usize;
    ///
    fn num_elem_1(&self) -> usize;
    ///
    fn num_elem_2(&self) -> usize;

    ///
    fn new_simplex_0(&self) -> Self::Simplex0;
    ///
    fn new_simplex_1(&self) -> Self::Simplex1;
    ///
    fn new_simplex_2(&self) -> Self::Simplex2;

    /// Discrete exterior derivative operator for primal 0-forms.
    ///
    /// The operator maps primal 0-forms to primal 1-forms.
    fn derivative_0_primal(&self, d_src: &mut Self::Simplex1, src: &Self::Simplex0) {
        <Self as DerivativePrimal<T, Self::Simplex0, Self::Simplex1>>::apply(self, d_src, src)
    }

    fn derivative_0_dual(&self, d_src: &mut Self::Simplex1, src: &Self::Simplex2) {
        <Self as DerivativeDual<T, Self::Simplex2, Self::Simplex1>>::apply(self, d_src, src)
    }

    /// Discrete exterior derivative operator for primal 1-forms.
    ///
    /// The operator maps primal 1-forms to primal 2-forms.
    fn derivative_1_primal(&self, d_src: &mut Self::Simplex2, src: &Self::Simplex1) {
        <Self as DerivativePrimal<T, Self::Simplex1, Self::Simplex2>>::apply(self, d_src, src)
    }
    fn derivative_1_dual(&self, d_src: &mut Self::Simplex0, src: &Self::Simplex1) {
        <Self as DerivativeDual<T, Self::Simplex1, Self::Simplex0>>::apply(self, d_src, src)
    }

    fn hodge_0_primal(&self, dual: &mut Self::Simplex0, primal: &Self::Simplex0) {
        <Self as Hodge<T, Self::Simplex0>>::apply(self, dual, primal)
    }
    fn hodge_2_dual(&self, primal: &mut Self::Simplex0, dual: &Self::Simplex0) {
        <Self as Hodge<T, Self::Simplex0>>::apply_inv(self, primal, dual)
    }

    fn hodge_1_primal(&self, dual: &mut Self::Simplex1, primal: &Self::Simplex1) {
        <Self as Hodge<T, Self::Simplex1>>::apply(self, dual, primal)
    }
    fn hodge_1_dual(&self, primal: &mut Self::Simplex1, dual: &Self::Simplex1) {
        <Self as Hodge<T, Self::Simplex1>>::apply_inv(self, primal, dual)
    }

    fn hodge_2_primal(&self, dual: &mut Self::Simplex2, primal: &Self::Simplex2) {
        <Self as Hodge<T, Self::Simplex2>>::apply(self, dual, primal)
    }
    fn hodge_0_dual(&self, primal: &mut Self::Simplex2, dual: &Self::Simplex2) {
        <Self as Hodge<T, Self::Simplex2>>::apply_inv(self, primal, dual)
    }
}

pub trait Hodge<T, Simplex> {
    fn apply(&self, dual: &mut Simplex, primal: &Simplex);
    fn apply_matrix(&self, &mut SparseMatrix<T>);
    fn apply_inv(&self, primal: &mut Simplex, dual: &Simplex);
    fn apply_matrix_inv(&self, &mut SparseMatrix<T>);
}

struct HodgeOperator<'a, T, M: 'a> {
    manifold: &'a M,
    _marker: PhantomData<T>,
}

pub struct Hodge0<'a, T, M: 'a>(HodgeOperator<'a, T, M>);
pub struct Hodge1<'a, T, M: 'a>(HodgeOperator<'a, T, M>);
pub struct Hodge2<'a, T, M: 'a>(HodgeOperator<'a, T, M>);

impl<'a, T, M> Mul<Hodge0<'a, T, M>> for SparseMatrix<T>
where
    M: DecDomain2d<T> + Hodge<T, <M as DecDomain2d<T>>::Simplex0>
{
    type Output = SparseMatrix<T>;
    fn mul(mut self, op: Hodge0<T, M>) -> SparseMatrix<T> {
        <M as Hodge<T, <M as DecDomain2d<T>>::Simplex0>>::apply_matrix(op.0.manifold, &mut self);
        self
    }
}

impl<'a, T, M> Mul<Hodge1<'a, T, M>> for SparseMatrix<T>
where
    M: DecDomain2d<T> + Hodge<T, <M as DecDomain2d<T>>::Simplex1>
{
    type Output = SparseMatrix<T>;
    fn mul(mut self, op: Hodge1<T, M>) -> SparseMatrix<T> {
        <M as Hodge<T, <M as DecDomain2d<T>>::Simplex1>>::apply_matrix(op.0.manifold, &mut self);
        self
    }
}

impl<'a, T, M> Mul<Hodge2<'a, T, M>> for SparseMatrix<T>
where
    M: DecDomain2d<T> + Hodge<T, <M as DecDomain2d<T>>::Simplex2>
{
    type Output = SparseMatrix<T>;
    fn mul(mut self, op: Hodge2<T, M>) -> SparseMatrix<T> {
        <M as Hodge<T, <M as DecDomain2d<T>>::Simplex2>>::apply_matrix(op.0.manifold, &mut self);
        self
    }
}

pub trait DerivativePrimal<T, S, DS> {
    fn apply(&self, &mut DS, &S);
    // fn apply_matrix(&self, &mut SparseMatrix<T>);
}

pub trait DerivativeDual<T, S, DS> {
    fn apply(&self, &mut DS, &S);
    // fn apply_matrix(&self, &mut SparseMatrix<T>);
}
