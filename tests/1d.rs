
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_fftw;

use ndarray::prelude::*;
use ndarray_fftw::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, Ix1>, b: Array<f64, Ix1>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nVectors not equal:\na = \n{:?}\nb = \n{:?}\n", a, b);
    }
}

#[test]
fn c2r2c() {
    let dist = Range::new(0., 1.);
    let a = Array::<c64, _>::random(65, dist);
}
