
use ndarray::{Array, Ix1};
use ffi::*;
use traits::*;
use num_traits::Zero;

trait Zeros: Sized {
    fn zeros(n: usize) -> Self;
}

impl<A: Zero + Clone> Zeros for Vec<A> {
    fn zeros(n: usize) -> Self {
        vec![A::zero(); n]
    }
}

macro_rules! impl_1d {
    ($RealArray:ty, $ComplexArray:ty, $r2r:path, $r2c:path, $c2r:path, $c2c:path) => {

impl RealArray for $RealArray {
    type ComplexArray = $ComplexArray;
    fn r2r(&self, flag: R2RKind) -> Self {
        let n = self.len();
        let mut out = Self::zeros(n);
        unsafe {
            let plan = $r2r(n as i32, self.as_ptr() as *mut _, out.as_mut_ptr(), flag.into(), FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
    fn r2c(&self) -> Self::ComplexArray {
        let n = self.len();
        let mut out = Self::ComplexArray::zeros(1 + n / 2);
        unsafe {
            let plan = $r2c(n as i32, self.as_ptr() as *mut _, out.as_mut_ptr() as *mut _, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
}

impl ComplexArray for $ComplexArray {
    type RealArray = $RealArray;
    fn c2r(&self) -> Self::RealArray {
        let n = (self.len() - 1) * 2;
        let mut out = Self::RealArray::zeros(n);
        unsafe {
            let plan = $c2r(n as i32, self.as_ptr() as *mut _, out.as_mut_ptr(), FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }

    fn c2c(&self, dir: C2CDirection) -> Self {
        let n = self.len();
        let mut out = Self::zeros(n);
        unsafe {
            let plan = $c2c(n as i32, self.as_ptr() as *mut _, out.as_mut_ptr() as *mut _, dir.into(), FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
}

}} // macro_rules impl_1d

impl_1d!(Vec<f64>, Vec<c64>, fftw_plan_r2r_1d, fftw_plan_dft_r2c_1d, fftw_plan_dft_c2r_1d, fftw_plan_dft_1d);
impl_1d!(Array<f64, Ix1>, Array<c64, Ix1>, fftw_plan_r2r_1d, fftw_plan_dft_r2c_1d, fftw_plan_dft_c2r_1d, fftw_plan_dft_1d);
