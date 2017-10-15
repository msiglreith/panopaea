
use math::{Dim, VectorN};

pub trait NeighborSearch<S, N>: Sync
where
    N: Dim<S>,
{
    fn build(&mut self, positons: &[VectorN<S, N>]);

    fn get_cell(&self, position: &VectorN<S, N>) -> Option<VectorN<usize, N>>;

    /// Apply function to each neighbor within a certain bound
    fn for_each_neighbor<F>(&self, cell: VectorN<usize, N>, bound: usize, fnc: F)
    where
        F: FnMut(usize);
}
