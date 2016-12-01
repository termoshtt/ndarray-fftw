use num_complex::Complex;
use ffi::*;
use types::*;

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;
#[allow(non_camel_case_types)]
type c32 = Complex<f32>;

pub trait RealArray {
    type ComplexArray;
    fn r2r(&self, flag: R2RKind) -> Self;
    fn r2c(&self) -> Self::ComplexArray;
}

pub trait ComplexArray {
    type RealArray;
    fn c2r(&self) -> Self::RealArray;
    fn c2c(&self, dir: C2CDirection) -> Self;
}

impl RealArray for Vec<f64> {
    type ComplexArray = Vec<c64>;

    fn r2r(&self, flag: R2RKind) -> Vec<f64> {
        let n = self.len();
        let mut out = vec![0.0; n];
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

    fn r2c(&self) -> Vec<c64> {
        let n = self.len();
        let mut out = vec![c64::new(0.0, 0.0); 1+n/2];
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

impl ComplexArray for Vec<c64> {
    type RealArray = Vec<f64>;
    fn c2r(&self) -> Vec<f64> {
        let n = (self.len() - 1) * 2;
        let mut out = vec![0.0; n];
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

    fn c2c(&self, dir: C2CDirection) -> Vec<c64> {
        let n = self.len();
        let mut out = vec![c64::new(0.0, 0.0); n];
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
