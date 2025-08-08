use crate::{Data, Error, Result, utils::bool_to_int};
use derive_ex::derive_ex;
use std::{
    ffi::c_void,
    fmt::Debug,
    ops::{Bound, Index, IndexMut, Range, RangeBounds},
    os::raw::c_int,
};
use text_grid::{cell, cells_schema, to_grid_with_schema};

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
pub struct MatBuf<T, L = ColMajor> {
    values: Vec<T>,
    nrow: usize,
    ncol: usize,
    layout: L,
}

impl<T, L: MatLayout> MatBuf<T, L> {
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
    #[track_caller]
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
impl<T> MatBuf<T, RowMajor> {
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
impl<T> MatBuf<T, ColMajor> {
    #[track_caller]
    pub fn col(&self, col: usize) -> &[T] {
        assert_col(col, self.ncol);
        &self.values[col * self.nrow..][..self.nrow]
    }
    #[track_caller]
    pub fn col_mut(&mut self, col: usize) -> &mut [T] {
        assert_col(col, self.ncol);
        &mut self.values[col * self.nrow..][..self.nrow]
    }
    pub fn cols(&self, range: impl RangeBounds<usize>) -> Mat<'_, T, ColMajor> {
        self.as_mat().cols(range)
    }
}
impl<T> MatBuf<T, RowMajor> {
    #[track_caller]
    pub fn row(&self, row: usize) -> &[T] {
        assert_row(row, self.nrow);
        &self.values[row * self.ncol..][..self.ncol]
    }
    #[track_caller]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert_row(row, self.nrow);
        &mut self.values[row * self.ncol..][..self.ncol]
    }
    #[track_caller]
    pub fn rows(&self, range: impl RangeBounds<usize>) -> Mat<'_, T, RowMajor> {
        self.as_mat().rows(range)
    }
}

impl<T, L: MatLayout> MatBuf<T, L> {
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
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> MatBuf<U, L> {
        MatBuf {
            values: self.values.iter().map(f).collect(),
            nrow: self.nrow,
            ncol: self.ncol,
            layout: self.layout,
        }
    }
    pub fn layout(&self) -> MatLayouts {
        self.layout.layout()
    }
}
impl<T, L: MatLayout> Index<[usize; 2]> for MatBuf<T, L> {
    type Output = T;
    #[track_caller]
    fn index(&self, [row, col]: [usize; 2]) -> &Self::Output {
        assert_row(row, self.nrow);
        assert_col(col, self.ncol);
        &self.values[self.layout.to_index(row, col, self.nrow, self.ncol)]
    }
}
impl<T, L: MatLayout> IndexMut<[usize; 2]> for MatBuf<T, L> {
    #[track_caller]
    fn index_mut(&mut self, [row, col]: [usize; 2]) -> &mut Self::Output {
        assert_row(row, self.nrow);
        assert_col(col, self.ncol);
        &mut self.values[self.layout.to_index(row, col, self.nrow, self.ncol)]
    }
}

impl<T: Debug> Debug for MatBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_mat().fmt(f)
    }
}
impl<T, L: MatLayout> AsMat<T> for MatBuf<T, L> {
    type Layout = L;
    fn as_mat(&self) -> Mat<'_, T, L> {
        Mat {
            values: &self.values,
            nrow: self.nrow,
            ncol: self.ncol,
            layout: self.layout,
        }
    }
}

#[derive_ex(Clone, Copy)]
pub struct Mat<'a, T, L: MatLayout> {
    values: &'a [T],
    nrow: usize,
    ncol: usize,
    layout: L,
}
impl<'a, T> Mat<'a, T, RowMajor> {
    pub fn from_row(row: &'a [T]) -> Self {
        Self::from_slice(row, 1, row.len(), RowMajor)
    }
}

impl<'a, T, L: MatLayout> Mat<'a, T, L> {
    #[track_caller]
    pub fn from_slice(values: &'a [T], nrow: usize, ncol: usize, layout: L) -> Self {
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

    pub fn nrow(&self) -> usize {
        self.nrow
    }
    pub fn ncol(&self) -> usize {
        self.ncol
    }
    pub fn layout(&self) -> MatLayouts {
        self.layout.layout()
    }
    pub fn as_slice(&self) -> &[T] {
        self.values
    }
    pub fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }
    pub(crate) fn as_data_ptr(&self) -> *const c_void
    where
        T: Data,
    {
        T::as_data_ptr(self.as_ptr())
    }

    pub(crate) fn is_row_major(&self) -> c_int {
        bool_to_int(self.layout() == MatLayouts::RowMajor)
    }
}
impl<'a, T> Mat<'a, T, ColMajor> {
    #[track_caller]
    pub fn col(&self, col: usize) -> &'a [T] {
        assert_col(col, self.ncol);
        &self.values[col * self.nrow..][..self.ncol]
    }

    #[track_caller]
    pub fn cols(&self, range: impl RangeBounds<usize>) -> Self {
        let range = to_range(range, self.ncol);
        assert_range(&range, self.ncol);
        let ncol = range.end - range.start;
        Self {
            values: &self.values[range.start * self.nrow..][..ncol * self.nrow],
            ncol,
            ..*self
        }
    }
}
impl<'a, T> Mat<'a, T, RowMajor> {
    #[track_caller]
    pub fn row(&self, row: usize) -> &'a [T] {
        assert_row(row, self.nrow);
        &self.values[row * self.ncol..][..self.ncol]
    }

    #[track_caller]
    pub fn rows(&self, range: impl RangeBounds<usize>) -> Self {
        let range = to_range(range, self.nrow);
        assert_range(&range, self.nrow);
        let nrow = range.end - range.start;
        Self {
            values: &self.values[range.start * self.ncol..][..nrow * self.ncol],
            nrow,
            ..*self
        }
    }
}

impl<T, L: MatLayout> AsMat<T> for Mat<'_, T, L> {
    type Layout = L;
    fn as_mat(&self) -> Mat<'_, T, L> {
        *self
    }
}
impl<T, L: MatLayout> Index<[usize; 2]> for Mat<'_, T, L> {
    type Output = T;
    #[track_caller]
    fn index(&self, [row, col]: [usize; 2]) -> &Self::Output {
        assert_row(row, self.nrow);
        assert_col(col, self.ncol);
        &self.values[self.layout.to_index(row, col, self.nrow, self.ncol)]
    }
}

impl<T: Debug, L: MatLayout> Debug for Mat<'_, T, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = cells_schema::<usize>(|f| {
            for col in 0..self.ncol {
                f.column(col, |&row| cell!("{:?}", &self[[row, col]]))
            }
        });
        write!(f, "{}", to_grid_with_schema(0..self.nrow, s))
    }
}

/// A trait for borrow [`Mat`].
pub trait AsMat<T> {
    type Layout: MatLayout;
    fn as_mat(&self) -> Mat<'_, T, Self::Layout>;
}

impl<M: AsMat<T>, T> AsMat<T> for &M {
    type Layout = M::Layout;
    fn as_mat(&self) -> Mat<'_, T, Self::Layout> {
        (*self).as_mat()
    }
}

fn to_range(value: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match value.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start + 1,
        Bound::Unbounded => 0,
    };
    let end = match value.end_bound() {
        Bound::Included(&end) => end + 1,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    start..end
}

#[track_caller]
fn assert_row(row: usize, nrow: usize) {
    assert_index("row", row, "nrow", nrow);
}

#[track_caller]
fn assert_col(col: usize, ncol: usize) {
    assert_index("col", col, "ncol", ncol);
}

#[track_caller]
fn assert_index(index_name: &str, index: usize, len_name: &str, len: usize) {
    assert!(
        index < len,
        "index out of bounds: the {len_name} is {len} but the {index_name} is {index}",
    );
}

#[track_caller]
fn assert_range(range: &Range<usize>, len: usize) {
    assert!(
        range.start <= range.end,
        "range start must be less than or equal to end"
    );
    assert!(
        range.end <= len,
        "index out of bounds: the len is {} but the range is {}..{}",
        len,
        range.start,
        range.end,
    );
}
