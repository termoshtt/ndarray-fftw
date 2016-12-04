
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_fftw;
extern crate ndarray_numtest;

use ndarray::prelude::*;
use ndarray_fftw::prelude::*;
use ndarray_numtest::prelude::*;
use ndarray_rand::RandomExt;

#[test]
fn r2c2r() {
    let dist = NormalAny::<f64>::new(0., 1.);
    let a = Array::random(128, dist);
    let ac = a.r2c();
    let acc = ac.c2r();
    (acc / 128.0).assert_allclose(&a, 1.0e-7);
}

#[test]
fn c2r2c() {
    let dist = ComplexNormal::<f64>::new(1.0, 1.0, 1.0, 1.0);
    let a = Array::random(65, dist);
    let ac = a.c2r();
    let acc = ac.r2c();
    acc.assert_allclose(&a, 1.0e-7);
}
