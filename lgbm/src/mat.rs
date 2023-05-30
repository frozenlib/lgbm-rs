use crate::{utils::bool_to_int, Error, Result};
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    os::raw::c_int,
};
use text_grid::{cell, grid_schema, Grid};

pub trait MatLayout: Copy + Clone + Send + Sync + Debug {
    fn layout(&self) -> MatLayouts;

    #[inline]
    fn to_index(&self, row: usize, col: usize, nrow: usize, ncol: usize) -> usize {
        match self.layout() {
            MatLayouts::RowMajor => row * ncol + col,
            MatLayouts::ColMajor => col * nrow + row,
        }
    }
}

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub enum MatLayouts {
    RowMajor,
    #[default]
    ColMajor,
}
impl MatLayout for MatLayouts {
    #[inline]
    fn layout(&self) -> MatLayouts {
        *self
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct RowMajor;
impl MatLayout for RowMajor {
    #[inline]
    fn layout(&self) -> MatLayouts {
        MatLayouts::RowMajor
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct ColMajor;
impl MatLayout for ColMajor {
    #[inline]
    fn layout(&self) -> MatLayouts {
        MatLayouts::ColMajor
    }
}

/// Matrix with rows for data and columns for features
#[derive(Clone)]
pub struct Mat<T, L = ColMajor> {
    values: Vec<T>,
    nrow: usize,
    ncol: usize,
    layout: L,
}

impl<T, L: MatLayout> Mat<T, L> {
    pub fn new(nrow: usize, ncol: usize) -> Self
    where
        T: Default,
        L: Default,
    {
        let mut values = Vec::new();
        values.resize_with(nrow * ncol, Default::default);
        Self {
            values,
            nrow,
            ncol,
            layout: Default::default(),
        }
    }
    pub fn from_vec(values: Vec<T>, nrow: usize, ncol: usize, layout: L) -> Self {
        if values.len() != nrow * ncol {
            panic!(
                "mismatch length : ({nrow} * {ncol} = {}, values.len() = {})",
                nrow * ncol,
                values.len()
            );
        }
        Self {
            values,
            nrow,
            ncol,
            layout,
        }
    }
}
impl<T> Mat<T, RowMajor> {
    pub fn from_rows<const N: usize>(rows: impl IntoIterator<Item = [T; N]>) -> Self {
        let mut nrow = 0;
        let mut values = Vec::new();
        for row in rows {
            values.extend(row);
            nrow += 1;
        }
        Self {
            values,
            nrow,
            ncol: N,
            layout: RowMajor,
        }
    }
    pub fn from_rows_non_empty<R: AsRef<[T]>>(
        rows: impl IntoIterator<Item = R>,
    ) -> Result<Option<Self>>
    where
        T: Clone,
    {
        let mut nrow = 0;
        let mut ncolumn = None;
        let mut values = Vec::new();
        for row in rows {
            let row = row.as_ref();
            if let Some(ncolumn) = ncolumn {
                if ncolumn != row.len() {
                    return Err(Error::from_message("mismatch column length"));
                }
            } else {
                ncolumn = Some(row.len());
            }
            for value in row {
                values.push(value.clone());
            }
            nrow += 1;
        }
        if let Some(ncolumn) = ncolumn {
            Ok(Some(Self {
                values,
                nrow,
                ncol: ncolumn,
                layout: RowMajor,
            }))
        } else {
            Ok(None)
        }
    }
}
impl<T> Mat<T, ColMajor> {
    pub fn col(&self, col: usize) -> &[T] {
        &self.values[col * self.nrow..][..self.ncol]
    }
    pub fn col_mut(&mut self, col: usize) -> &mut [T] {
        &mut self.values[col * self.nrow..][..self.ncol]
    }
}
impl<T> Mat<T, RowMajor> {
    pub fn row(&self, row: usize) -> &[T] {
        &self.values[row * self.ncol..][..self.ncol]
    }
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        &mut self.values[row * self.ncol..][..self.ncol]
    }
}

impl<T, L: MatLayout> Mat<T, L> {
    pub fn nrow(&self) -> usize {
        self.nrow
    }
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    pub fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.values.as_mut_ptr()
    }
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Mat<U, L> {
        Mat {
            values: self.values.iter().map(f).collect(),
            nrow: self.nrow,
            ncol: self.ncol,
            layout: self.layout,
        }
    }
    pub fn layout(&self) -> MatLayouts {
        self.layout.layout()
    }
    pub(crate) fn is_row_major(&self) -> c_int {
        bool_to_int(self.layout() == MatLayouts::RowMajor)
    }
}
impl<T, L: MatLayout> Index<[usize; 2]> for Mat<T, L> {
    type Output = T;
    fn index(&self, [row, col]: [usize; 2]) -> &Self::Output {
        &self.values[self.layout.to_index(row, col, self.nrow, self.ncol)]
    }
}
impl<T, L: MatLayout> IndexMut<[usize; 2]> for Mat<T, L> {
    fn index_mut(&mut self, [row, col]: [usize; 2]) -> &mut Self::Output {
        &mut self.values[self.layout.to_index(row, col, self.nrow, self.ncol)]
    }
}

impl<T: Debug> Debug for Mat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = grid_schema::<usize>(|f| {
            for col in 0..self.ncol {
                f.column(col, |&&row| cell!("{:?}", &self[[row, col]]))
            }
        });
        let mut g = Grid::new_with_schema(s);
        g.extend(0..self.nrow);
        write!(f, "g")
    }
}
