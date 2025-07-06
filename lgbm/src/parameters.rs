//! <https://lightgbm.readthedocs.io/en/latest/Parameters.html>

use crate::{Result, utils::to_cstring};
use parse_display::{Display, FromStr};
use serde::{Deserialize, Serialize};
use std::ffi::CString;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    None,
    Bool(bool),
    Int(i64),
    USize(usize),
    Float(f64),
    String(String),
    Array(Vec<ParameterValue>),
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterValue::None => write!(f, "None"),
            ParameterValue::Bool(value) => write!(f, "{value}"),
            ParameterValue::Int(value) => write!(f, "{value}"),
            ParameterValue::USize(value) => write!(f, "{value}"),
            ParameterValue::Float(value) => write!(f, "{value}"),
            ParameterValue::String(value) => write!(f, "{value}"),
            ParameterValue::Array(value) => {
                if value.is_empty() {
                    write!(f, "None")
                } else {
                    for (i, value) in value.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{value}")?;
                    }
                    Ok(())
                }
            }
        }
    }
}
impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<i32> for ParameterValue {
    fn from(value: i32) -> Self {
        Self::Int(value.into())
    }
}
impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}
impl From<usize> for ParameterValue {
    fn from(value: usize) -> Self {
        Self::USize(value)
    }
}
impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}
impl From<&'static str> for ParameterValue {
    fn from(value: &'static str) -> Self {
        Self::String(value.to_string())
    }
}
impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl<T: Into<ParameterValue>> From<Vec<T>> for ParameterValue {
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<ParameterValue>, const N: usize> From<[T; N]> for ParameterValue {
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html>
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Parameters(pub Vec<(String, ParameterValue)>);

impl Parameters {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn push(&mut self, key: impl Into<String>, value: impl Into<ParameterValue>) {
        self.0.push((key.into(), value.into()));
    }
}

impl std::fmt::Display for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, (key, value)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{key}={value}")?;
        }
        Ok(())
    }
}

impl Parameters {
    pub fn to_cstring(&self) -> Result<CString> {
        to_cstring(&self.to_string())
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective>
#[derive(
    Copy,
    Clone,
    Default,
    Debug,
    PartialEq,
    Eq,
    Display,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum Objective {
    #[default]
    Regression,
    RegressionL1,
    Huber,
    Fair,
    Poisson,
    Quantile,
    Mape,
    Gamma,
    Tweedie,
    Binary,
    Multiclass,
    Multiclassova,
    CrossEntropy,
    CrossEntropyLambda,
    Lambdarank,
    RankXendcg,
    Custom,
}
impl From<Objective> for ParameterValue {
    fn from(value: Objective) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#boosting>
#[derive(
    Copy,
    Clone,
    Default,
    Debug,
    PartialEq,
    Eq,
    Display,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum Boosting {
    #[default]
    Gbdt,
    Rf,
    Dart,
    Goss,
}
impl From<Boosting> for ParameterValue {
    fn from(value: Boosting) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#data_sample_strategy>
#[derive(
    Copy,
    Clone,
    Default,
    Debug,
    PartialEq,
    Eq,
    Display,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum DataSampleStrategy {
    #[default]
    Bagging,
    Goss,
}
impl From<DataSampleStrategy> for ParameterValue {
    fn from(value: DataSampleStrategy) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#tree_learner>
#[derive(
    Copy,
    Clone,
    Default,
    Debug,
    PartialEq,
    Eq,
    Display,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum TreeLearner {
    #[default]
    Serial,
    Feature,
    Data,
    Voting,
}
impl From<TreeLearner> for ParameterValue {
    fn from(value: TreeLearner) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#device_type>
#[derive(
    Copy,
    Clone,
    Default,
    Debug,
    PartialEq,
    Eq,
    Display,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum DeviceType {
    #[default]
    Cpu,
    Gpu,
    Cuda,
}

impl From<DeviceType> for ParameterValue {
    fn from(value: DeviceType) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric>
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Display,
    FromStr,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[display(style = "snake_case")]
pub enum Metric {
    L1,
    L2,
    Rmse,
    Quantile,
    Mape,
    Huber,
    Fair,
    Poisson,
    Gamma,
    GammaDeviance,
    Tweedie,
    Ndcg,
    Map,
    Auc,
    AveragePrecision,
    BinaryLogloss,
    BinaryError,
    AucMu,
    MultiLogloss,
    MultiError,
    CrossEntropy,
    CrossEntropyLambda,
    KullbackLeibler,
}

impl Metric {
    pub fn is_lower_is_better(self) -> bool {
        match self {
            Metric::L1 => true,
            Metric::L2 => true,
            Metric::Rmse => true,
            Metric::Quantile => true,
            Metric::Mape => true,
            Metric::Huber => true,
            Metric::Fair => true,
            Metric::Poisson => true,
            Metric::Gamma => true,
            Metric::GammaDeviance => true,
            Metric::Tweedie => true,
            Metric::Ndcg => false,
            Metric::Map => false,
            Metric::Auc => false,
            Metric::AveragePrecision => false,
            Metric::BinaryLogloss => true,
            Metric::BinaryError => true,
            Metric::AucMu => false,
            Metric::MultiLogloss => true,
            Metric::MultiError => true,
            Metric::CrossEntropy => true,
            Metric::CrossEntropyLambda => true,
            Metric::KullbackLeibler => true,
        }
    }
}

impl From<Metric> for ParameterValue {
    fn from(value: Metric) -> Self {
        value.to_string().into()
    }
}

/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#verbosity>
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, Display, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]

pub enum Verbosity {
    Fatal,
    Error,
    Info,
    Debug,
}
impl From<Verbosity> for ParameterValue {
    fn from(value: Verbosity) -> Self {
        match value {
            Verbosity::Fatal => ParameterValue::Int(-1),
            Verbosity::Error => ParameterValue::Int(0),
            Verbosity::Info => ParameterValue::Int(1),
            Verbosity::Debug => ParameterValue::Int(2),
        }
    }
}

/*
/// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters>
#[derive(Clone, Debug, PartialEq)]
#[derive_ex(Default)]
pub struct CoreParameters {
    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective>
    pub objective: Objective,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#boosting>
    pub boosting: Boosting,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#data_sample_strategy>
    pub data_sample_strategy: DataSampleStrategy,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#learning_rate>
    #[default(0.1)]
    pub learning_rate: f64,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#num_leaves>
    #[default(31)]
    pub num_leaves: i32,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#tree_learner>
    pub tree_learner: TreeLearner,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#num_threads>
    #[default(0)]
    pub num_threads: i32,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#device_type>
    pub device_type: DeviceType,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#seed>
    pub seed: Option<i32>,

    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic>
    pub deterministic: bool,
    // c-api seems to ignore num_iterations
    // #[default(100)]
    // pub num_iterations: i32,
}

#[derive(Clone, Debug, PartialEq)]
#[derive_ex(Default)]
pub struct LearningControlParameters {
    #[default(false)]
    pub force_col_wise: bool,

    #[default(false)]
    pub force_row_wise: bool,

    #[default(-1.0)]
    pub histogram_pool_size: f64,

    #[default(-1)]
    pub max_depth: i32,

    #[default(20)]
    pub min_data_in_leaf: i32,

    #[default(1e-3)]
    pub min_sum_hessian_in_leaf: f64,

    #[default(1.0)]
    pub bagging_fraction: f64,

    #[default(1.0)]
    pub pos_bagging_fraction: f64,

    #[default(1.0)]
    pub neg_bagging_fraction: f64,

    #[default(0)]
    pub bagging_freq: i32,

    #[default(3)]
    pub bagging_seed: i32,
    // todo
}
*/
