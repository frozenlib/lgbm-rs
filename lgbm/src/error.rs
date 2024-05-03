use lgbm_sys::LGBM_GetLastError;
use std::{
    ffi::{CStr, FromBytesWithNulError, FromVecWithNulError, IntoStringError, NulError},
    num::TryFromIntError,
    os::raw::c_int,
    string::FromUtf8Error,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    code: Option<c_int>,
    message: String,
}

impl Error {
    pub fn from_message(message: &str) -> Self {
        Self {
            code: None,
            message: message.to_string(),
        }
    }
    pub fn from_error(e: impl std::error::Error) -> Self {
        Self {
            code: None,
            message: e.to_string(),
        }
    }
    pub fn code(&self) -> Option<c_int> {
        self.code
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for Error {}

pub(crate) fn to_result(code: c_int) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        unsafe {
            let message = CStr::from_ptr(LGBM_GetLastError())
                .to_string_lossy()
                .into_owned();
            Err(Error {
                code: Some(code),
                message,
            })
        }
    }
}

impl From<TryFromIntError> for Error {
    fn from(value: TryFromIntError) -> Self {
        Self::from_error(value)
    }
}
impl From<NulError> for Error {
    fn from(value: NulError) -> Self {
        Self::from_error(value)
    }
}
impl From<FromBytesWithNulError> for Error {
    fn from(value: FromBytesWithNulError) -> Self {
        Self::from_error(value)
    }
}
impl From<FromVecWithNulError> for Error {
    fn from(value: FromVecWithNulError) -> Self {
        Self::from_error(value)
    }
}
impl From<IntoStringError> for Error {
    fn from(value: IntoStringError) -> Self {
        Self::from_error(value)
    }
}
impl From<FromUtf8Error> for Error {
    fn from(value: FromUtf8Error) -> Self {
        Self::from_error(value)
    }
}
