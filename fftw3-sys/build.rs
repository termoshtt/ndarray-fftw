// Copied from https://github.com/huonw/fftw3-rs
// Under MIT License
// Copyright (c) 2014 Huon Wilson

extern crate pkg_config;
fn main() {
    match pkg_config::find_library("fftw3") {
        Ok(_) => return,
        Err(_) => {}
    }

    println!("failed to find libfftw3, and automatically building it is currently unimplemented.");
    std::process::exit(1);
}
