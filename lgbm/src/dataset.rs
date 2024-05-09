use crate::{
    mat::AsMat,
    to_result,
    utils::{get_strings, path_to_cstring, to_cstring},
    Error, Parameters, Result,
};
use lgbm_sys::{
    DatasetHandle, LGBM_DatasetCreateFromFile, LGBM_DatasetCreateFromMat,
    LGBM_DatasetCreateFromMats, LGBM_DatasetDumpText, LGBM_DatasetFree,
    LGBM_DatasetGetFeatureNames, LGBM_DatasetGetField, LGBM_DatasetGetNumData,
    LGBM_DatasetGetNumFeature, LGBM_DatasetSetFeatureNames, LGBM_DatasetSetField,
    C_API_DTYPE_FLOAT32, C_API_DTYPE_FLOAT64, C_API_DTYPE_INT32, C_API_DTYPE_INT64,
};
use std::{
    marker::PhantomData,
    os::raw::{c_int, c_void},
    path::Path,
    ptr::{null, null_mut},
    slice,
};

pub trait Data: Sized {
    const DATA_TYPE: c_int;
    fn as_data_ptr(data: *const Self) -> *const c_void;
}
impl Data for f32 {
    const DATA_TYPE: c_int = C_API_DTYPE_FLOAT32 as c_int;
    fn as_data_ptr(data: *const Self) -> *const c_void {
        data as *const c_void
    }
}
impl Data for f64 {
    const DATA_TYPE: c_int = C_API_DTYPE_FLOAT64 as c_int;
    fn as_data_ptr(data: *const Self) -> *const c_void {
        data as *const c_void
    }
}
impl Data for i32 {
    const DATA_TYPE: c_int = C_API_DTYPE_INT32 as c_int;
    fn as_data_ptr(data: *const Self) -> *const c_void {
        data as *const c_void
    }
}
impl Data for i64 {
    const DATA_TYPE: c_int = C_API_DTYPE_INT64 as c_int;
    fn as_data_ptr(data: *const Self) -> *const c_void {
        data as *const c_void
    }
}

pub trait FeatureData: Data {}

impl FeatureData for f32 {}
impl FeatureData for f64 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field<T> {
    name: &'static [u8],
    _type: PhantomData<T>,
}
impl<T> Field<T> {
    const fn new(name: &'static [u8]) -> Self {
        assert!(matches!(name.last(), Some(&0)));
        Self {
            name,
            _type: PhantomData,
        }
    }

    fn name_ptr(&self) -> *const i8 {
        self.name.as_ptr() as *const i8
    }
}
impl Field<f32> {
    pub const LABEL: Self = Self::new(b"label\0");
    pub const WEIGHT: Self = Self::new(b"weight\0");
}
impl Field<f64> {
    pub const INIT_SCORE: Self = Self::new(b"init_score\0");
}
impl Field<i32> {
    pub const GROUP: Self = Self::new(b"group\0");
}

/// Owned [DatasetHandle](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.DatasetHandle)
pub struct Dataset(pub(crate) DatasetHandle);

impl Dataset {
    /// [LGBM_DatasetCreateFromFile](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromFile)
    #[doc(alias = "LGBM_DatasetCreateFromFile")]
    pub fn from_file(
        filename: &Path,
        reference: Option<&Dataset>,
        parameters: &Parameters,
    ) -> Result<Self> {
        let mut handle = null_mut();
        unsafe {
            to_result(LGBM_DatasetCreateFromFile(
                path_to_cstring(filename)?.as_ptr(),
                parameters.to_cstring()?.as_ptr(),
                to_dataset_handle(reference),
                &mut handle,
            ))?;
        }
        Ok(Self(handle))
    }

    /// [LGBM_DatasetCreateFromMat](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromMat)
    #[doc(alias = "LGBM_DatasetCreateFromMat")]
    pub fn from_mat<T: FeatureData>(
        mat: impl AsMat<T>,
        reference: Option<&Dataset>,
        parameters: &Parameters,
    ) -> Result<Self> {
        let mat = mat.as_mat();
        let mut handle = null_mut();
        unsafe {
            to_result(LGBM_DatasetCreateFromMat(
                mat.as_data_ptr(),
                T::DATA_TYPE,
                mat.nrow().try_into()?,
                mat.ncol().try_into()?,
                mat.is_row_major(),
                parameters.to_cstring()?.as_ptr(),
                to_dataset_handle(reference),
                &mut handle,
            ))?;
        }
        Ok(Self(handle))
    }
    /// [LGBM_DatasetCreateFromMats](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromMats)
    #[doc(alias = "LGBM_DatasetCreateFromMats")]
    pub fn from_mats<M: AsMat<T>, T: FeatureData>(
        mats: impl IntoIterator<Item = M>,
        reference: Option<&Dataset>,
        parameters: &Parameters,
    ) -> Result<Self> {
        let as_mats = mats.into_iter().collect::<Vec<_>>();
        let mats = as_mats.iter().map(|x| x.as_mat()).collect::<Vec<_>>();
        if mats.is_empty() {
            return Err(Error::from_message("mats must not be empty"));
        }
        let ncol = mats[0].ncol();
        let is_row_major = mats[0].is_row_major();
        let mut nrows: Vec<i32> = Vec::with_capacity(mats.len());
        let mut mat_ptrs = Vec::with_capacity(mats.len());
        for mat in &mats {
            if mat.ncol() != ncol {
                return Err(Error::from_message(
                    "mats must have the same number of columns",
                ));
            }
            if mat.is_row_major() != is_row_major {
                return Err(Error::from_message("mats must have the same layout"));
            }
            nrows.push(mat.nrow().try_into()?);
            mat_ptrs.push(mat.as_data_ptr());
        }
        let mut handle = null_mut();
        unsafe {
            to_result(LGBM_DatasetCreateFromMats(
                mats.len().try_into()?,
                mat_ptrs.as_mut_ptr(),
                T::DATA_TYPE,
                nrows.as_mut_ptr(),
                ncol.try_into()?,
                is_row_major,
                parameters.to_cstring()?.as_ptr(),
                to_dataset_handle(reference),
                &mut handle,
            ))?;
        }
        Ok(Self(handle))
    }

    /// [LGBM_DatasetSetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField)
    #[doc(alias = "LGBM_DatasetSetField")]
    pub fn set_field<T: Data>(&mut self, field: Field<T>, data: &[T]) -> Result<()> {
        unsafe {
            to_result(LGBM_DatasetSetField(
                self.0,
                field.name_ptr(),
                data.as_ptr() as *const c_void,
                data.len().try_into()?,
                T::DATA_TYPE,
            ))
        }
    }

    /// [LGBM_DatasetGetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetField)
    #[doc(alias = "LGBM_DatasetGetField")]
    pub fn get_field<T: Data>(&self, field: Field<T>) -> Result<&[T]> {
        unsafe {
            let mut out_len = 0;
            let mut out_ptr = null();
            let mut out_type = 0;
            to_result(LGBM_DatasetGetField(
                self.0,
                field.name_ptr(),
                &mut out_len,
                &mut out_ptr,
                &mut out_type,
            ))?;
            if out_type != T::DATA_TYPE {
                return Err(Error::from_message("element type mismatch"));
            }
            Ok(slice::from_raw_parts(out_ptr as *const T, out_len as usize))
        }
    }

    /// [LGBM_DatasetGetNumFeature](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetNumFeature)
    #[doc(alias = "LGBM_DatasetGetNumFeature")]
    pub fn get_num_feature(&self) -> Result<usize> {
        let mut num_feature = 0;
        unsafe {
            to_result(LGBM_DatasetGetNumFeature(self.0, &mut num_feature))?;
        }
        Ok(num_feature as usize)
    }

    /// [LGBM_DatasetGetNumData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetNumData)
    #[doc(alias = "LGBM_DatasetGetNumData")]
    pub fn get_num_data(&self) -> Result<usize> {
        let mut num_data = 0;
        unsafe {
            to_result(LGBM_DatasetGetNumData(self.0, &mut num_data))?;
        }
        Ok(num_data as usize)
    }

    /// [LGBM_DatasetDumpText](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetDumpText)
    #[doc(alias = "LGBM_DatasetDumpText")]
    pub fn dump_text(&self, path: &Path) -> Result<()> {
        unsafe {
            to_result(LGBM_DatasetDumpText(
                self.0,
                path_to_cstring(path)?.as_ptr(),
            ))
        }
    }

    /// [LGBM_DatasetSetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetFeatureNames)
    #[doc(alias = "LGBM_DatasetSetFeatureNames")]
    pub fn set_feature_names<T: AsRef<str>>(
        &mut self,
        names: impl IntoIterator<Item = T>,
    ) -> Result<()> {
        let mut cstr_names = Vec::new();
        for name in names {
            cstr_names.push(to_cstring(name.as_ref())?);
        }
        let mut pcstr_names = cstr_names.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
        unsafe {
            to_result(LGBM_DatasetSetFeatureNames(
                self.0,
                pcstr_names.as_mut_ptr(),
                pcstr_names.len().try_into()?,
            ))
        }
    }

    /// [LGBM_DatasetGetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetFeatureNames)
    #[doc(alias = "LGBM_DatasetGetFeatureNames")]
    pub fn get_feature_names(&self) -> Result<Vec<String>> {
        get_strings(
            |len, out_len, buffer_len, out_buffer_len, out_strs| unsafe {
                LGBM_DatasetGetFeatureNames(
                    self.0,
                    len,
                    out_len,
                    buffer_len,
                    out_buffer_len,
                    out_strs,
                )
            },
        )
    }
}
impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe {
            to_result(LGBM_DatasetFree(self.0)).unwrap();
        }
    }
}
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

fn to_dataset_handle(dataset: Option<&Dataset>) -> DatasetHandle {
    if let Some(dataset) = dataset {
        dataset.0
    } else {
        null_mut()
    }
}
