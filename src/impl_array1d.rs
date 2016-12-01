
use ndarray::{Array, Ix1};
use ffi::*;
use traits::*;

impl RealArray for Array<f64, Ix1> {
    type ComplexArray = Array<c64, Ix1>;
    fn r2r(&self, flag: R2RKind) -> Self {
        let n = self.len();
        let mut out = Array::zeros(n);
        unsafe {
            let plan = fftw_plan_r2r_1d(n as i32,
                                        self.as_ptr() as *mut _,
                                        out.as_mut_ptr(),
                                        flag.into(),
                                        FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
    fn r2c(&self) -> Self::ComplexArray {
        let n = self.len();
        let mut out = Array::zeros(1 + n / 2);
        unsafe {
            let plan = fftw_plan_dft_r2c_1d(n as i32,
                                            self.as_ptr() as *mut _,
                                            out.as_mut_ptr() as *mut _,
                                            FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
}

impl ComplexArray for Array<c64, Ix1> {
    type RealArray = Array<f64, Ix1>;
    fn c2r(&self) -> Self::RealArray {
        let n = (self.len() - 1) * 2;
        let mut out = Array::zeros(n);
        unsafe {
            let plan = fftw_plan_dft_c2r_1d(n as i32,
                                            self.as_ptr() as *mut _,
                                            out.as_mut_ptr(),
                                            FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }

    fn c2c(&self, dir: C2CDirection) -> Self {
        let n = self.len();
        let mut out = Array::zeros(n);
        unsafe {
            let plan = fftw_plan_dft_1d(n as i32,
                                        self.as_ptr() as *mut _,
                                        out.as_mut_ptr() as *mut _,
                                        dir.into(),
                                        FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        out
    }
}
