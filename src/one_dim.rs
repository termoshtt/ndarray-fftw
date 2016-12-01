
use num_complex::Complex;
use ffi::*;
use types::*;

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;
#[allow(non_camel_case_types)]
type c32 = Complex<f32>;

pub fn r2r(mut in_: Vec<f64>, flag: R2RKind) -> Vec<f64> {
    let n = in_.len();
    let mut out = vec![0.0; n];
    unsafe {
        let plan = fftw_plan_r2r_1d(n as i32,
                                    in_.as_mut_ptr(),
                                    out.as_mut_ptr(),
                                    flag.into(),
                                    FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    out
}

pub fn r2c(mut in_: Vec<f64>) -> Vec<c64> {
    let n = in_.len();
    let mut out = vec![c64::new(0.0, 0.0); 1+n/2];
    unsafe {
        let plan = fftw_plan_dft_r2c_1d(n as i32,
                                        in_.as_mut_ptr(),
                                        out.as_mut_ptr() as *mut _,
                                        FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    out
}

pub fn c2r(mut in_: Vec<c64>) -> Vec<f64> {
    let n = (in_.len() - 1) * 2;
    let mut out = vec![0.0; n];
    unsafe {
        let plan = fftw_plan_dft_c2r_1d(n as i32,
                                        in_.as_mut_ptr() as *mut _,
                                        out.as_mut_ptr(),
                                        FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    out
}

pub fn c2c(mut in_: Vec<c64>, dir: C2CDirection) -> Vec<c64> {
    let n = in_.len();
    let mut out = vec![c64::new(0.0, 0.0); n];
    unsafe {
        let plan = fftw_plan_dft_1d(n as i32,
                                    in_.as_mut_ptr() as *mut _,
                                    out.as_mut_ptr() as *mut _,
                                    dir.into(),
                                    FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    out
}
