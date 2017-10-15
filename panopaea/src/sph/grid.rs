
//! Bounded Unfiform Grid

use generic_array::typenum::{U2, U3};
use math::{Dim, Real, VectorN};
use super::neighbor::NeighborSearch;
use num::Zero;
use std::usize;
use std::cmp;

pub struct BoundedGrid<S: Real, N: Dim<usize> + Dim<(usize, usize)>> {
    num_cells: VectorN<usize, N>,
    strides: VectorN<usize, N>,
    cell_size: S,
    cell_ranges: Vec<(usize, usize)>,
}

impl<N, S> BoundedGrid<S, N>
where
    S: Real,
    N: Dim<S> + Dim<i64> + Dim<(usize, usize)>,
{
    pub fn new(num_cells: VectorN<usize, N>, cell_size: S) -> Self {
        let num_dims = N::to_usize();
        let mut strides = VectorN::from_elem(1);
        for i in 1..num_dims {
            strides[i] = strides[i-1] * num_cells[i-1];
        }
        let total_cells = strides[num_dims-1] * num_cells[num_dims-1];
        let cell_ranges = vec![(0, 0); total_cells];

        BoundedGrid {
            num_cells,
            strides,
            cell_size,
            cell_ranges,
        }
    }

    pub fn get_key(&self, position: &VectorN<S, N>) -> usize {
        if let Some(cell) = self.get_cell(position) {
            (0..N::to_usize()).map(|i| cell[i]*self.strides[i]).sum()
        } else {
            usize::MAX
        }
    }

    pub fn get_cell(&self, position: &VectorN<S, N>) -> Option<VectorN<usize, N>> {
        let mut cell = VectorN::<i64, N>::zero();
        for i in 0..N::to_usize() {
            cell[i] = (position[i] / self.cell_size).floor().to_i64().unwrap()
        }

        if (0..N::to_usize()).all(|i| 0 <= cell[i] && cell[i] < self.num_cells[i] as i64) {
            let mut c = VectorN::zero();
            for i in 0..N::to_usize() {
                c[i] = cell[i] as usize;
            }

            Some(c)
        } else {
            None
        }
    }

    pub fn build(&mut self, positions: &[VectorN<S, N>]) {
        // reset ranges
        for cell in &mut self.cell_ranges {
            *cell = (0, 0);
        }

        let mut prev = self.get_key(&positions[0]);

        {
            if prev >= self.cell_ranges.len() { return; }
            self.cell_ranges[prev].0 = 0;
        }

        for particle in 1..positions.len() {
            let index = self.get_key(&positions[particle]);

            if index >= self.cell_ranges.len() {
                self.cell_ranges[prev].1 = particle;
                return;
            }

            if prev != index {
                // new cell
                self.cell_ranges[index].0 = particle;
                self.cell_ranges[prev].1 = particle;
            }

            prev = index;
        }

        self.cell_ranges[prev].1 = positions.len();
    }
}

impl<S> BoundedGrid<S, U2>
where
    S: Real,
{
    pub fn get_range(&self, cell: (usize, usize)) -> Option<(usize, usize)> {
        if (cell.0 < self.num_cells[0]) &&
           (cell.1 < self.num_cells[1])
        {
            Some(unsafe { self.get_range_unchecked(cell) })
        } else {
            println!("WARN!");
            None
        }
    }

    pub unsafe fn get_range_unchecked(&self, cell: (usize, usize)) -> (usize, usize) {
        let index = cell.0 + cell.1 * self.num_cells[0];
        self.cell_ranges[index]
    }
}

impl<S> NeighborSearch<S, U2> for BoundedGrid<S, U2>
where
    S: Real,
{
    fn build(&mut self, positions: &[VectorN<S, U2>]) {
        self.build(positions)
    }

    fn get_cell(&self, position: &VectorN<S, U2>) -> Option<VectorN<usize, U2>> {
        self.get_cell(position)
    }

    fn for_each_neighbor<F>(&self, cell: VectorN<usize, U2>, bound: usize, mut fnc: F)
        where F: FnMut(usize)
    {
        let upper_x = cmp::min(cell[0] + bound+1, self.num_cells[0]);
        let upper_y = cmp::min(cell[1] + bound+1, self.num_cells[1]);

        for y in cell[1].saturating_sub(bound)..upper_y {
            for x in cell[0].saturating_sub(bound)..upper_x {
                let (start, end) = unsafe { self.get_range_unchecked((x, y)) };
                assert!(start <= end);
                for p in start..end {
                    fnc(p);
                }
            }
        }
    }
}

impl<S> NeighborSearch<S, U3> for BoundedGrid<S, U3>
where
    S: Real,
{
    fn build(&mut self, positions: &[VectorN<S, U3>]) {
        self.build(positions)
    }

    fn get_cell(&self, position: &VectorN<S, U3>) -> Option<VectorN<usize, U3>> {
        self.get_cell(position)
    }

    fn for_each_neighbor<F>(&self, cell: VectorN<usize, U3>, bound: usize, fnc: F)
        where F: FnMut(usize)
    {
        let upper_x = cmp::min(cell[0] + bound+1, self.num_cells[0]);
        let upper_y = cmp::min(cell[1] + bound+1, self.num_cells[1]);
        let upper_z = cmp::min(cell[2] + bound+1, self.num_cells[2]);

        for z in cell[2].saturating_sub(bound)..upper_z {
            for y in cell[1].saturating_sub(bound)..upper_y {
                for x in cell[0].saturating_sub(bound)..upper_x {
                    unimplemented!()
                }
            }
        }
    }
}

