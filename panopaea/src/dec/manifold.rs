
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
    // fn apply_matrix(&self, &mut SparseMatrix<T>);
    fn apply_inv(&self, primal: &mut Simplex, dual: &Simplex);
    // fn apply_matrix_inv(&self, &mut SparseMatrix<T>);
}

pub struct Hodge0<'a, T, S, M: 'a + Hodge<T, S>> {
    manifold: &'a M,
    _marker: PhantomData<(S, T)>,
}

/*
impl<'a, S, T, M> Mul<Hodge0<'a, T, S, M>> for SparseMatrix<T>
where
    M: Hodge<T, S>
{
    type Output = SparseMatrix<T>;
    fn mul(self, op: Hodge0<T, S, M>) -> SparseMatrix<T> {
        <M as Hodge<T, S>>::apply_matrix(op.manifold, &mut self);
        self
    }
}
*/

pub trait DerivativePrimal<T, S, DS> {
    fn apply(&self, &mut DS, &S);
    // fn apply_matrix(&self, &mut SparseMatrix<T>);
}

pub trait DerivativeDual<T, S, DS> {
    fn apply(&self, &mut DS, &S);
    // fn apply_matrix(&self, &mut SparseMatrix<T>);
}
