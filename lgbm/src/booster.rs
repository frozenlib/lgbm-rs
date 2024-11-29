use crate::{
    mat::AsMat,
    to_result,
    utils::{get_cstring, get_strings, int_to_bool, path_to_cstring},
    Dataset, Error, FeatureData, Parameters, Result,
};
use lgbm_sys::{
    BoosterHandle, LGBM_BoosterAddValidData, LGBM_BoosterCalcNumPredict, LGBM_BoosterCreate,
    LGBM_BoosterCreateFromModelfile, LGBM_BoosterDumpModel, LGBM_BoosterFeatureImportance,
    LGBM_BoosterFree, LGBM_BoosterGetCurrentIteration, LGBM_BoosterGetEval,
    LGBM_BoosterGetEvalCounts, LGBM_BoosterGetEvalNames, LGBM_BoosterGetFeatureNames,
    LGBM_BoosterGetNumClasses, LGBM_BoosterGetNumFeature, LGBM_BoosterGetNumPredict,
    LGBM_BoosterGetPredict, LGBM_BoosterLoadModelFromString, LGBM_BoosterNumModelPerIteration,
    LGBM_BoosterNumberOfTotalModel, LGBM_BoosterPredictForMat, LGBM_BoosterRollbackOneIter,
    LGBM_BoosterSaveModel, LGBM_BoosterSaveModelToString, LGBM_BoosterUpdateOneIter,
    LGBM_BoosterUpdateOneIterCustom, C_API_FEATURE_IMPORTANCE_GAIN, C_API_FEATURE_IMPORTANCE_SPLIT,
    C_API_MATRIX_TYPE_CSC, C_API_MATRIX_TYPE_CSR, C_API_PREDICT_CONTRIB, C_API_PREDICT_LEAF_INDEX,
    C_API_PREDICT_NORMAL, C_API_PREDICT_RAW_SCORE,
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::{CStr, CString},
    os::raw::c_int,
    path::Path,
    ptr::null_mut,
    sync::Arc,
};
use text_grid::{cells_f, cells_schema, to_grid_with_schema, Cells};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictType {
    Normal = C_API_PREDICT_NORMAL,
    RawScore = C_API_PREDICT_RAW_SCORE,
    LeafIndex = C_API_PREDICT_LEAF_INDEX,
    Contrib = C_API_PREDICT_CONTRIB,
}
impl PredictType {
    fn to_cint(self) -> c_int {
        self as u32 as c_int
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixType {
    Csr = C_API_MATRIX_TYPE_CSR,
    Csc = C_API_MATRIX_TYPE_CSC,
}
impl MatrixType {
    // fn to_cint(self) -> c_int {
    //     self as u32 as c_int
    // }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureImportanceType {
    Split = C_API_FEATURE_IMPORTANCE_SPLIT,
    Gain = C_API_FEATURE_IMPORTANCE_GAIN,
}
impl FeatureImportanceType {
    fn to_cint(self) -> c_int {
        self as u32 as c_int
    }
}

/// Owned [BoosterHandle](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.BoosterHandle)
pub struct Booster {
    handle: BoosterHandle,
    data: Vec<Option<Arc<Dataset>>>,
}

impl Booster {
    fn from_handle(handle: BoosterHandle, train_data: Option<Arc<Dataset>>) -> Self {
        Self {
            handle,
            data: vec![train_data],
        }
    }

    /// [LGBM_BoosterCreate](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCreate)
    #[doc(alias = "LGBM_BoosterCreate")]
    pub fn new(train_data: Arc<Dataset>, parameters: &Parameters) -> Result<Self> {
        let mut handle: BoosterHandle = null_mut();
        unsafe {
            to_result(LGBM_BoosterCreate(
                train_data.0,
                parameters.to_cstring()?.as_ptr(),
                &mut handle,
            ))?;
        }
        Ok(Self::from_handle(handle, Some(train_data)))
    }

    /// [LGBM_BoosterCreateFromModelfile](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCreateFromModelfile)
    #[doc(alias = "LGBM_BoosterCreateFromModelfile")]
    pub fn from_file(filename: &Path) -> Result<(Self, usize)> {
        let mut handle = null_mut();
        let mut out_num_iterations = 0;
        unsafe {
            to_result(LGBM_BoosterCreateFromModelfile(
                path_to_cstring(filename)?.as_ptr(),
                &mut out_num_iterations,
                &mut handle,
            ))?;
        }
        Ok((Self::from_handle(handle, None), out_num_iterations as usize))
    }
    /// [LGBM_BoosterLoadModelFromString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterLoadModelFromString)
    #[doc(alias = "LGBM_BoosterLoadModelFromString")]
    pub fn from_string(model: &CStr) -> Result<(Self, usize)> {
        let mut handle = null_mut();
        let mut out_num_iterations = 0;
        unsafe {
            to_result(LGBM_BoosterLoadModelFromString(
                model.as_ptr(),
                &mut out_num_iterations,
                &mut handle,
            ))?;
        }
        Ok((Self::from_handle(handle, None), out_num_iterations as usize))
    }

    pub fn data(&self, data_idx: usize) -> Option<&Arc<Dataset>> {
        self.data.get(data_idx).and_then(|x| x.as_ref())
    }

    /// [LGBM_BoosterAddValidData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterAddValidData)
    #[doc(alias = "LGBM_BoosterAddValidData")]
    pub fn add_valid_data(&mut self, dataset: Arc<Dataset>) -> Result<()> {
        unsafe { to_result(LGBM_BoosterAddValidData(self.handle, dataset.0))? }
        self.data.push(Some(dataset));
        Ok(())
    }

    /// [LGBM_BoosterCalcNumPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCalcNumPredict)
    #[doc(alias = "LGBM_BoosterCalcNumPredict")]
    pub fn calc_num_predict(
        &self,
        num_row: usize,
        predict_type: PredictType,
        start_iteration: usize,
        num_iteration: Option<usize>,
    ) -> Result<usize> {
        let num_row = num_row.try_into()?;
        let predict_type = predict_type.to_cint();
        let start_iteration = start_iteration.try_into()?;
        let num_iteration = num_iteration.unwrap_or(0).try_into()?;
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterCalcNumPredict(
                self.handle,
                num_row,
                predict_type,
                start_iteration,
                num_iteration,
                &mut out_len,
            ))?;
        }
        Ok(out_len as usize)
    }

    /// [LGBM_BoosterDumpModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterDumpModel)
    #[doc(alias = "LGBM_BoosterDumpModel")]
    pub fn dump_model(
        &self,
        start_iteration: usize,
        num_iteration: Option<usize>,
        feature_importance_type: FeatureImportanceType,
    ) -> Result<CString> {
        let start_iteration = start_iteration.try_into()?;
        let num_iteration = num_iteration.unwrap_or(0).try_into()?;
        let feature_importance_type = feature_importance_type.to_cint();
        get_cstring(move |buffer_len, out_len, out_str| unsafe {
            LGBM_BoosterDumpModel(
                self.handle,
                start_iteration,
                num_iteration,
                feature_importance_type,
                buffer_len,
                out_len,
                out_str,
            )
        })
    }

    /// [LGBM_BoosterFeatureImportance](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterFeatureImportance)
    #[doc(alias = "LGBM_BoosterFeatureImportance")]
    pub fn feature_importance(
        &self,
        num_iteration: Option<usize>,
        importance_type: FeatureImportanceType,
    ) -> Result<Vec<f64>> {
        let num_iteration = num_iteration.unwrap_or(0).try_into()?;
        let mut out_results = vec![f64::NAN; self.get_num_feature()?];
        unsafe {
            to_result(LGBM_BoosterFeatureImportance(
                self.handle,
                num_iteration,
                importance_type.to_cint(),
                out_results.as_mut_ptr(),
            ))?;
        }
        Ok(out_results)
    }

    /// [LGBM_BoosterGetCurrentIteration](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetCurrentIteration)
    #[doc(alias = "LGBM_BoosterGetCurrentIteration")]
    pub fn get_current_iteration(&self) -> Result<usize> {
        let mut out_iter = 0;
        unsafe {
            to_result(LGBM_BoosterGetCurrentIteration(self.handle, &mut out_iter))?;
        }
        Ok(out_iter as usize)
    }

    /// [LGBM_BoosterGetEval](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEval)
    #[doc(alias = "LGBM_BoosterGetEval")]
    pub fn get_eval(&self, data_idx: usize) -> Result<Vec<f64>> {
        let mut out_results = Vec::with_capacity(self.get_eval_counts()?);
        unsafe {
            let mut out_len = 0;
            to_result(LGBM_BoosterGetEval(
                self.handle,
                data_idx.try_into()?,
                &mut out_len,
                out_results.as_mut_ptr(),
            ))?;
            out_results.set_len(out_len as usize);
        }
        Ok(out_results)
    }

    /// [LGBM_BoosterGetEvalCounts](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEvalCounts)
    #[doc(alias = "LGBM_BoosterGetEvalCounts")]
    fn get_eval_counts(&self) -> Result<usize> {
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterGetEvalCounts(self.handle, &mut out_len))?;
        }
        Ok(out_len as usize)
    }

    /// [LGBM_BoosterGetEvalNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEvalNames)
    #[doc(alias = "LGBM_BoosterGetEvalNames")]
    pub fn get_eval_names(&self) -> Result<Vec<String>> {
        get_strings(
            |len, out_len, buffer_len, out_buffer_len, out_strs| unsafe {
                LGBM_BoosterGetEvalNames(
                    self.handle,
                    len,
                    out_len,
                    buffer_len,
                    out_buffer_len,
                    out_strs,
                )
            },
        )
    }

    /// [LGBM_BoosterGetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetFeatureNames)
    #[doc(alias = "LGBM_BoosterGetFeatureNames")]
    pub fn get_feature_names(&self) -> Result<Vec<String>> {
        get_strings(
            |len, out_len, buffer_len, out_buffer_len, out_strs| unsafe {
                LGBM_BoosterGetFeatureNames(
                    self.handle,
                    len,
                    out_len,
                    buffer_len,
                    out_buffer_len,
                    out_strs,
                )
            },
        )
    }

    /// [LGBM_BoosterGetNumClasses](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumClasses)
    #[doc(alias = "LGBM_BoosterGetNumClasses")]
    pub fn get_num_classes(&self) -> Result<usize> {
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterGetNumClasses(self.handle, &mut out_len))?;
        }
        Ok(out_len as usize)
    }

    /// [LGBM_BoosterGetNumFeature](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumFeature)
    #[doc(alias = "LGBM_BoosterGetNumFeature")]
    pub fn get_num_feature(&self) -> Result<usize> {
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterGetNumFeature(self.handle, &mut out_len))?;
        }
        Ok(out_len as usize)
    }

    /// [LGBM_BoosterGetNumPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumPredict)
    #[doc(alias = "LGBM_BoosterGetNumPredict")]
    pub fn get_num_predict(&self, data_idx: usize) -> Result<usize> {
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterGetNumPredict(
                self.handle,
                data_idx.try_into()?,
                &mut out_len,
            ))?;
        }
        Ok(out_len as usize)
    }

    fn get_num_data(&self, data_idx: usize) -> Result<usize> {
        if let Some(Some(data)) = self.data.get(data_idx) {
            data.get_num_data()
        } else {
            Err(Error::from_message("invlaid data_idx"))
        }
    }

    /// [LGBM_BoosterNumberOfTotalModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterNumberOfTotalModel)
    pub fn number_of_total_model(&self) -> Result<usize> {
        let mut value = 0;
        unsafe {
            to_result(LGBM_BoosterNumberOfTotalModel(self.handle, &mut value))?;
        }
        Ok(value as usize)
    }

    /// [LGBM_BoosterNumModelPerIteration](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterNumModelPerIteration)
    pub fn num_model_per_iteration(&self) -> Result<usize> {
        let mut value = 0;
        unsafe {
            to_result(LGBM_BoosterNumModelPerIteration(self.handle, &mut value))?;
        }
        Ok(value as usize)
    }

    /// [LGBM_BoosterGetPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetPredict)
    #[doc(alias = "LGBM_BoosterGetPredict")]
    pub fn get_predict(&self, data_idx: usize) -> Result<Prediction> {
        let num_data = self.get_num_data(data_idx)?;
        let num_predict = self.get_num_predict(data_idx)?;
        let num_class = self.get_num_classes()?;
        let mut out_result = Prediction::from_num_predict(num_predict, num_data, num_class)?;
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterGetPredict(
                self.handle,
                data_idx.try_into()?,
                &mut out_len,
                out_result.values.as_mut_ptr(),
            ))?;
        }
        assert!(out_len as usize == out_result.values.len());
        Ok(out_result)
    }

    /// [LGBM_BoosterPredictForMat](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMat)
    #[doc(alias = "LGBM_BoosterPredictForMat")]
    pub fn predict_for_mat<T: FeatureData>(
        &self,
        mat: impl AsMat<T>,
        predict_type: PredictType,
        start_iteration: usize,
        num_iteration: Option<usize>,
        parameters: &Parameters,
    ) -> Result<Prediction> {
        let mat = mat.as_mat();
        let num_feature = self.get_num_feature()?;
        if num_feature != mat.ncol() {
            return Err(Error::from_message(&format!(
                "column size must be {num_feature}, but got {}",
                mat.ncol(),
            )));
        }
        let num_row = mat.nrow();
        let num_class = self.get_num_classes()?;
        let num_predict =
            self.calc_num_predict(num_row, predict_type, start_iteration, num_iteration)?;
        let num_iteration = num_iteration.unwrap_or(0).try_into()?;

        let mut out_result = Prediction::from_num_predict(num_predict, num_row, num_class)?;
        let mut out_len = 0;
        unsafe {
            to_result(LGBM_BoosterPredictForMat(
                self.handle,
                T::as_data_ptr(mat.as_ptr()),
                T::DATA_TYPE,
                mat.nrow().try_into()?,
                mat.ncol().try_into()?,
                mat.is_row_major(),
                predict_type.to_cint(),
                start_iteration.try_into()?,
                num_iteration,
                parameters.to_cstring()?.as_ptr(),
                &mut out_len,
                out_result.values.as_mut_ptr(),
            ))?;
        }
        assert_eq!(out_len as usize, out_result.values.len());
        Ok(out_result)
    }

    /// [LGBM_BoosterUpdateOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIter)
    #[doc(alias = "LGBM_BoosterUpdateOneIter")]
    pub fn update_one_iter(&mut self) -> Result<bool> {
        let mut is_finished = 0;
        unsafe {
            to_result(LGBM_BoosterUpdateOneIter(self.handle, &mut is_finished))?;
        }
        Ok(int_to_bool(is_finished))
    }

    /// [LGBM_BoosterUpdateOneIterCustom](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIterCustom)
    #[doc(alias = "LGBM_BoosterUpdateOneIterCustom")]
    pub fn update_one_iter_custom(&mut self, grad: &[f32], hess: &[f32]) -> Result<bool> {
        let num_class = self.get_num_classes()?;
        let num_data = self.get_num_data(0)?;
        assert_eq!(grad.len(), num_class * num_data, "mismatch grad length.");
        assert_eq!(hess.len(), num_class * num_data, "mismatch hess length.");
        let mut is_finished = 0;
        unsafe {
            to_result(LGBM_BoosterUpdateOneIterCustom(
                self.handle,
                grad.as_ptr(),
                hess.as_ptr(),
                &mut is_finished,
            ))?;
        }
        Ok(int_to_bool(is_finished))
    }

    /// [LGBM_BoosterRollbackOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterRollbackOneIter)
    #[doc(alias = "LGBM_BoosterRollbackOneIter")]
    pub fn rollback_one_iter(&mut self) -> Result<()> {
        unsafe { to_result(LGBM_BoosterRollbackOneIter(self.handle)) }
    }

    /// [LGBM_BoosterSaveModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModel)
    #[doc(alias = "LGBM_BoosterSaveModel")]
    pub fn save_model(
        &self,
        start_iteration: usize,
        num_iteration: Option<usize>,
        feature_importance_type: FeatureImportanceType,
        filename: &Path,
    ) -> Result<()> {
        unsafe {
            to_result(LGBM_BoosterSaveModel(
                self.handle,
                start_iteration.try_into()?,
                num_iteration.unwrap_or(0).try_into()?,
                feature_importance_type.to_cint(),
                path_to_cstring(filename)?.as_ptr(),
            ))?;
        }
        Ok(())
    }

    /// [LGBM_BoosterSaveModelToString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModelToString)
    #[doc(alias = "LGBM_BoosterSaveModelToString")]
    pub fn save_model_to_string(
        &self,
        start_iteration: usize,
        num_iteration: Option<usize>,
        feature_importance_type: FeatureImportanceType,
    ) -> Result<CString> {
        let start_iteration = start_iteration.try_into()?;
        let num_iteration = num_iteration.unwrap_or(0).try_into()?;
        let feature_importance_type = feature_importance_type.to_cint();
        get_cstring(move |buffer_len, out_len, out_str| unsafe {
            LGBM_BoosterSaveModelToString(
                self.handle,
                start_iteration,
                num_iteration,
                feature_importance_type,
                buffer_len,
                out_len,
                out_str,
            )
        })
    }
}
impl Drop for Booster {
    fn drop(&mut self) {
        unsafe {
            to_result(LGBM_BoosterFree(self.handle)).unwrap();
        }
    }
}
unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

#[derive(Clone, Serialize, Deserialize)]
pub struct Prediction {
    num: [usize; 3],
    values: Vec<f64>,
}
impl Prediction {
    fn new(num_data: usize, num_class: usize, num_2: usize) -> Self {
        Self {
            values: vec![f64::NAN; num_data * num_class * num_2],
            num: [num_data, num_class, num_2],
        }
    }
    fn from_num_predict(num_predict: usize, num_data: usize, num_class: usize) -> Result<Self> {
        if num_data * num_class == 0 {
            return Ok(Self::new(num_data, num_class, 1));
        }
        let num_2 = num_predict / num_class / num_data;
        if num_data * num_class * num_2 != num_predict {
            return Err(Error::from_message("invalid num_data"));
        }
        Ok(Self::new(num_data, num_class, num_2))
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn num_data(&self) -> usize {
        self.num[0]
    }
    pub fn num_class(&self) -> usize {
        self.num[1]
    }
    pub fn num_iteration(&self) -> usize {
        self.num[2]
    }
    pub fn num_feature(&self) -> usize {
        self.num[2] - 1
    }

    fn fmt_with<T: Cells>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        to_cells: impl Fn(f64) -> T,
    ) -> std::fmt::Result {
        writeln!(f, "num_data  : {}", self.num_data())?;
        writeln!(f, "num_class : {}", self.num_class())?;
        writeln!(f, "num_2     : {}", self.num[2])?;
        self.fmt_values_with(f, to_cells)
    }
    fn fmt_values_with<T: Cells>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        to_cells: impl Fn(f64) -> T,
    ) -> std::fmt::Result {
        writeln!(f)?;
        match (self.num_class(), self.num[2]) {
            (_, 1) => {
                let schema = cells_schema(|f| {
                    f.column("", |&row| row);
                    for column in 0..self.num_class() {
                        f.column(column, |&row| to_cells(self[[row, column]]));
                    }
                });
                writeln!(f, "{}", to_grid_with_schema(0..self.num_data(), schema))?;
            }
            (_, _) => {
                writeln!(f, "{:?}", &self.values)?;
            }
        }
        Ok(())
    }

    pub fn approx_eq(&self, other: &Self, margin: f64) -> bool {
        self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(&v0, &v1)| approx_eq(v0, v1, margin))
    }
}
fn approx_eq(v0: f64, v1: f64, margin: f64) -> bool {
    // `v0 == v1` to return true for the same infinity.
    v0 == v1 || (v0 - v1).abs() <= margin
}

impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = f.precision();
        if let Some(p) = p {
            self.fmt_with(f, |x| cells_f!("{:.*}", p, x))
        } else {
            self.fmt_with(f, |x| cells_f!("{:}", x))
        }
    }
}
impl std::fmt::LowerExp for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = f.precision();
        if let Some(p) = p {
            self.fmt_with(f, |x| cells_f!("{:.*e}", p, x))
        } else {
            self.fmt_with(f, |x| cells_f!("{:e}", x))
        }
    }
}
impl std::fmt::UpperExp for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = f.precision();
        if let Some(p) = p {
            self.fmt_with(f, |x| cells_f!("{:.*E}", p, x))
        } else {
            self.fmt_with(f, |x| cells_f!("{:E}", x))
        }
    }
}
impl std::fmt::Debug for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Values<'a>(&'a Prediction);
        impl std::fmt::Debug for Values<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let p = f.precision();
                if let Some(p) = p {
                    self.0.fmt_values_with(f, |x| cells_f!("{:.*?}", p, x))
                } else {
                    self.0.fmt_values_with(f, |x| cells_f!("{:?}", x))
                }
            }
        }
        f.debug_struct("Prediction")
            .field("num", &self.num)
            .field("values", &Values(self))
            .finish()
    }
}

impl std::ops::Index<usize> for Prediction {
    type Output = f64;
    fn index(&self, data: usize) -> &f64 {
        assert_eq!(self.num_class(), 1, "num_class");
        assert_eq!(self.num[2], 1, "num_2");
        &self.values[data]
    }
}
impl std::ops::Index<[usize; 2]> for Prediction {
    type Output = f64;
    fn index(&self, [data, class]: [usize; 2]) -> &f64 {
        assert_eq!(self.num[2], 1, "num_2");
        &self.values[data * self.num_class() + class]
    }
}
impl std::ops::Index<[usize; 3]> for Prediction {
    type Output = f64;
    fn index(&self, [data, class, iteration]: [usize; 3]) -> &f64 {
        &self.values[data * self.num_class() * self.num[2] + class * self.num[2] + iteration]
    }
}

impl From<Prediction> for Vec<f64> {
    fn from(p: Prediction) -> Self {
        p.values
    }
}
