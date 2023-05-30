fn main() -> anyhow::Result<()> {
    #[derive(Debug)]
    struct Callbacks;

    impl bindgen::callbacks::ParseCallbacks for Callbacks {
        fn process_comment(&self, comment: &str) -> Option<String> {
            Some(doxygen_rs::transform(comment))
        }
    }

    let bindings = bindgen::Builder::default()
        .clang_arg("-Isubmodules/LightGBM/include")
        .header("submodules/LightGBM/include/LightGBM/c_api.h")
        .parse_callbacks(Box::new(Callbacks))
        .allowlist_file("submodules/LightGBM/include/LightGBM/c_api.h")
        .generate()?;

    bindings.write_to_file("lgbm-sys/src/lib.rs")?;
    Ok(())
}
