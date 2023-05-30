use crate::{to_result, Error, Result};
use std::{
    ffi::{c_char, CString},
    os::raw::c_int,
    path::Path,
    ptr::null_mut,
};

pub fn to_cstring(value: &str) -> Result<CString> {
    if let Ok(value) = CString::new(value) {
        Ok(value)
    } else {
        Err(convert_string_error())
    }
}

pub const fn bool_to_int(value: bool) -> c_int {
    if value {
        1
    } else {
        0
    }
}
pub const fn int_to_bool(value: c_int) -> bool {
    value != 0
}

pub fn path_to_cstring(path: &Path) -> Result<CString> {
    if let Some(s) = path.to_str() {
        return Ok(CString::new(s)?);
    }
    Err(convert_string_error())
}
pub fn get_strings(
    get: impl Fn(c_int, *mut c_int, usize, *mut usize, *mut *mut c_char) -> c_int,
) -> Result<Vec<String>> {
    let mut len = 0;
    let mut buffer_len = 0;
    to_result(get(0, &mut len, 0, &mut buffer_len, null_mut()))?;
    let mut buffers = vec![vec![0u8; buffer_len]; len as usize];
    let mut out_strs = buffers
        .iter_mut()
        .map(|x| x.as_mut_ptr() as *mut c_char)
        .collect::<Vec<_>>();
    let mut out_len = 0;
    let mut out_buffer_len = 0;
    to_result(get(
        len,
        &mut out_len,
        buffer_len,
        &mut out_buffer_len,
        out_strs.as_mut_ptr(),
    ))?;
    out_strs.truncate(out_len as usize);
    buffers.into_iter().map(chars_to_string).collect()
}
pub fn get_bytes(get: impl Fn(i64, *mut i64, *mut c_char) -> c_int) -> Result<Vec<u8>> {
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        let mut out_len = 0;
        to_result(get(
            buffer.len() as i64,
            &mut out_len,
            buffer.as_mut_ptr() as *mut c_char,
        ))?;
        let out_len = out_len as usize;
        if out_len <= buffer.len() {
            buffer.truncate(out_len);
            return Ok(buffer);
        }
        buffer.resize(out_len, 0);
    }
}
pub fn get_cstring(get: impl Fn(i64, *mut i64, *mut c_char) -> c_int) -> Result<CString> {
    Ok(CString::from_vec_with_nul(get_bytes(get)?)?)
}

pub fn chars_to_string(mut chars: Vec<u8>) -> Result<String> {
    if let Some(index) = chars.iter().position(|&x| x == 0) {
        chars.truncate(index);
    }
    Ok(String::from_utf8(chars)?)
}

fn convert_string_error() -> Error {
    Error::from_message("failed to convert string")
}
