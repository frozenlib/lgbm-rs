use std::{env, path::Path};

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let os = std::env::consts::OS;
    match os {
        "windows" => build_windows(),
        "linux" => build_linux(),
        "macos" => build_macos(),
        _ => panic!("Unsupported OS: {}", os),
    }
}
fn build_windows() {
    let dir = env_var("LIGHTGBM_LIB_DIR");
    println!("cargo:rustc-link-search={dir}");
    let dir_path = Path::new(&dir);
    let lib_path = dir_path.join("lib_lightgbm.lib");
    let dll_path = dir_path.join("lib_lightgbm.dll");
    rerun_if_changed(&lib_path);
    rerun_if_changed(&dll_path);

    if !lib_path.is_file() {
        panic!("lib_lightgbm.lib not found in {dir}");
    }
    if dll_path.is_file() {
        println!("cargo:rustc-link-lib=dylib=lib_lightgbm");
    } else {
        println!("cargo:rustc-link-lib=static=lib_lightgbm");
    }
}
fn build_linux() {
    let dir = env_var("LIGHTGBM_LIB_DIR");
    println!("cargo:rustc-link-search={dir}");
    let dir_path = Path::new(&dir);
    let a_path = dir_path.join("lib_lightgbm.a");
    let so_path = dir_path.join("lib_lightgbm.so");
    rerun_if_changed(&a_path);
    rerun_if_changed(&so_path);
    if a_path.is_file() {
        println!("cargo:rustc-link-lib=static=_lightgbm");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    } else if so_path.is_file() {
        println!("cargo:rustc-link-lib=dylib=_lightgbm");
    } else {
        panic!("both lib_lightgbm.a and lib_lightgbm.so not found in {dir}");
    }
}
fn build_macos() {
    let target = env::var("TARGET").unwrap();
    if target.contains("aarch64") {
        println!("cargo:rustc-link-lib=dylib=lightgbm");
    } else {
        println!("cargo:rustc-link-lib=dylib=_lightgbm");
    }
}

fn env_var(key: &str) -> String {
    println!("cargo:rerun-if-env-changed={key}");
    env::var(key).unwrap_or_else(|_| panic!("environment variable `{key}` is not set"))
}
fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}
