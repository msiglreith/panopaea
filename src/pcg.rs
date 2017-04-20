
use math::{LinearView, LinearViewReal, MulOut, Real};

pub trait Preconditioner<L> {
    fn apply(&self, dst: &mut L, src: &L);
}

impl<A: Clone, L: LinearView<Elem = A>> Preconditioner<L> for () {
    fn apply(&self, dst: &mut L, src: &L) {
        dst.view_linear_mut().assign(&src.view_linear());
    }
}

pub fn precond_conjugate_gradient<L, O, P, T>(
    preconditioner: &P,
    x: &mut L,
    b: &L,
    max_iterations: usize,
    threshold: T,
    residual: &mut L,
    mut auxiliary: &mut L,
    search: &mut L,
    mut A: O,
) where P: Preconditioner<L>,
        T: Real,
        L: LinearViewReal<T>,
        O: FnMut(&mut L, &L),
{ 
    // Conjugate gradient

    // initial guess
    x.view_linear_mut().fill(T::zero());

    // early out
    if b.norm_max() < threshold {
        println!("b start norm: {:?}", b.norm_max());
        return;
    }

    residual.view_linear_mut().assign(&b.view_linear());
    preconditioner.apply(auxiliary, residual);
    search.view_linear_mut().assign(&auxiliary.view_linear());

    {
        let mut residual_error = T::zero();
        let mut sigma = auxiliary.dot_linear(residual);

        'iter: for i in 0..max_iterations {
            // println!("residual: {:#?}", &residual.view_linear());
            // println!("search: {:#?}", &search.view_linear());
            A(&mut auxiliary, search); // apply_sparse_matrix(auxiliary, search, diag, plus_x, plus_y, timestep);
            // println!("aux: {:#?}", &auxiliary.view_linear());
            let alpha = sigma/auxiliary.dot_linear(search);
            
            x.view_linear_mut().scaled_add(alpha, &search.view_linear());
            residual.view_linear_mut().scaled_add(-alpha, &auxiliary.view_linear());

            residual_error = residual.norm_max();
            // println!("{:?}", residual_error);
            if residual_error < threshold {
                println!("Iterations {:?}", i);
                break 'iter;
            }

            preconditioner.apply(auxiliary, residual);
            
            let sigma_new = auxiliary.dot_linear(residual);
            let beta = sigma_new/sigma;

            // println!("beta: {:#?}", beta);

            let mut search = search.view_linear_mut();
            let auxiliary = auxiliary.view_linear();

            for s in 0..search.len() {
                search[s] = auxiliary[s] + beta*search[s];
            }

            sigma = sigma_new;
        }
    }
}
