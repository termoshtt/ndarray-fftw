
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_fftw;

use ndarray::prelude::*;
use ndarray_fftw::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(test: Array<f64, Ix1>, truth: Array<f64, Ix1>) {
    if !test.all_close(&truth, 1.0e-7) {
        panic!("\nVectors not equal:\ntest = \n{:?}\ntruth = \n{:?}\n",
               test,
               truth);
    }
}

#[test]
fn r2c2r() {
    let dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random(128, dist);
    let ac = a.r2c();
    let acc = ac.c2r();
    all_close(acc / 128.0, a);
}
