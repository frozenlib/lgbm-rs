use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let os = std::env::consts::OS;
    match os {
        "windows" => build_windows(),
        "linux" => build_linux(),
        "macos" => build_macos(),
        _ => panic!("Unsupported OS: {os}"),
    }
}
fn build_windows() {
    let dir = env_var("LIGHTGBM_LIB_DIR");
    println!("cargo:rustc-link-search={dir}");
    let dir_path = Path::new(&dir);
    let lib_path = dir_path.join("lib_lightgbm.lib");
    let dll_path = dir_path.join("lib_lightgbm.dll");
    rerun_if_changed(dir_path);
    if !lib_path.is_file() {
        panic!("lib_lightgbm.lib not found in {dir}");
    }
    if dll_path.is_file() {
        println!("cargo:rustc-link-lib=dylib=lib_lightgbm");
    } else {
        println!("cargo:rustc-link-lib=static=lib_lightgbm");
        link_nanoarrow(dir_path);
    }
}
fn build_linux() {
    let dir = env_var("LIGHTGBM_LIB_DIR");
    println!("cargo:rustc-link-search={dir}");
    let dir_path = Path::new(&dir);
    let a_path = dir_path.join("lib_lightgbm.a");
    let so_path = dir_path.join("lib_lightgbm.so");
    rerun_if_changed(dir_path);
    if a_path.is_file() {
        println!("cargo:rustc-link-lib=static=_lightgbm");
        link_nanoarrow(dir_path);
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    } else if so_path.is_file() {
        println!("cargo:rustc-link-lib=dylib=_lightgbm");
    } else {
        panic!("both lib_lightgbm.a and lib_lightgbm.so not found in {dir}");
    }
}
fn build_macos() {
    if let Some(dir) = try_env_var("LIGHTGBM_LIB_DIR") {
        println!("cargo:rustc-link-search={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=_lightgbm");
}

fn link_nanoarrow(lightgbm_lib_dir: &Path) {
    let lib_file_name = if cfg!(windows) {
        "nanoarrow_static.lib"
    } else {
        "libnanoarrow_static.a"
    };
    let mut candidates = Vec::new();
    if let Some(dir) = try_env_var("NANOARROW_LIB_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    candidates.push(lightgbm_lib_dir.to_path_buf());
    candidates.push(
        lightgbm_lib_dir
            .join("build")
            .join("external_libs")
            .join("nanoarrow"),
    );
    if let Some(parent) = lightgbm_lib_dir.parent() {
        candidates.push(parent.join("build").join("external_libs").join("nanoarrow"));
    }
    let nanoarrow_lib_dir = candidates
        .iter()
        .flat_map(|dir| [dir.clone(), dir.join("Release")])
        .find(|dir| dir.join(lib_file_name).is_file())
        .unwrap_or_else(|| {
            panic!(
                "{lib_file_name} not found; set NANOARROW_LIB_DIR to the directory containing it"
            )
        });
    println!("cargo:rustc-link-search={}", nanoarrow_lib_dir.display());
    println!("cargo:rustc-link-lib=static=nanoarrow_static");
}

fn env_var(key: &str) -> String {
    try_env_var(key).unwrap_or_else(|| panic!("environment variable `{key}` is not set"))
}
fn try_env_var(key: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={key}");
    env::var(key).ok()
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}
