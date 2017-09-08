
use sparse::{SparseMatrix};
use std::marker::PhantomData;
use std::ops::Mul;

/// Storage type for 0-simplices (vertex).
/// A differential primal 0-form is stored for each 0-simplex.
/// Storage type for 1-simplices (edge).
/// A differential primal 1-form is stored for each 1-simplex.
/// Storage type for 2-simplices (face).
/// A differential primal 2-form is stored for each 2-simplex.

pub trait Manifold2d<T, S0, S1, S2>
    : Hodge<T, S0> + Hodge<T, S1> + Hodge<T, S2>
    + DerivativePrimal<T, S0, S1> + DerivativeDual<T, S2, S1>
    + DerivativePrimal<T, S1, S2> + DerivativeDual<T, S1, S0>
{
    ///
    fn num_elem_0(&self) -> usize;
    ///
    fn num_elem_1(&self) -> usize;
    ///
    fn num_elem_2(&self) -> usize;

    ///
    fn new_simplex_0(&self) -> S0;
    ///
    fn new_simplex_1(&self) -> S1;
    ///
    fn new_simplex_2(&self) -> S2;

    /// Discrete exterior derivative operator for primal 0-forms.
    ///
    /// The operator maps primal 0-forms to primal 1-forms.
    fn derivative_0_primal(&self, &mut S1, &S0);
    fn derivative_0_dual(&self, &mut S1, &S2);

    /// Discrete exterior derivative operator for primal 1-forms.
    ///
    /// The operator maps primal 1-forms to primal 2-forms.
    fn derivative_1_primal(&self, &mut S2, &S1);
    fn derivative_1_dual(&self, &mut S0, &S1);

    fn hodge_0_primal(&self, dual: &mut S0, primal: &S0) {
        <Self as Hodge<T, S0>>::apply(self, dual, primal)
    }
    fn hodge_2_dual(&self, primal: &mut S0, dual: &S0) {
        <Self as Hodge<T, S0>>::apply_inv(self, primal, dual)
    }

    fn hodge_1_primal(&self, dual: &mut S1, primal: &S1) {
        <Self as Hodge<T, S1>>::apply(self, dual, primal)
    }
    fn hodge_1_dual(&self, primal: &mut S1, dual: &S1) {
        <Self as Hodge<T, S1>>::apply_inv(self, primal, dual)
    }

    fn hodge_2_primal(&self, dual: &mut S2, primal: &S2) {
        <Self as Hodge<T, S2>>::apply(self, dual, primal)
    }
    fn hodge_0_dual(&self, primal: &mut S2, dual: &S2) {
        <Self as Hodge<T, S2>>::apply_inv(self, primal, dual)
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
