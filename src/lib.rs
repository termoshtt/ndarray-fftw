
extern crate ndarray;
extern crate fftw3_sys as ffi;
extern crate num_complex;

use ffi::*;

#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex<f64>;

#[derive(Clone, Copy, Debug)]
pub enum R2RKind {
    R2HC,
    HC2R,
    DHT,
    DCT00,
    DCT01,
    DCT10,
    DCT11,
    DST00,
    DST01,
    DST10,
    DST11,
}

impl Into<u32> for R2RKind {
    fn into(self) -> u32 {
        match self {
            R2RKind::R2HC => ffi::FFTW_R2HC,
            R2RKind::HC2R => ffi::FFTW_HC2R,
            R2RKind::DHT => ffi::FFTW_DHT,
            R2RKind::DCT00 => ffi::FFTW_REDFT00,
            R2RKind::DCT01 => ffi::FFTW_REDFT01,
            R2RKind::DCT10 => ffi::FFTW_REDFT10,
            R2RKind::DCT11 => ffi::FFTW_REDFT11,
            R2RKind::DST00 => ffi::FFTW_RODFT00,
            R2RKind::DST01 => ffi::FFTW_RODFT01,
            R2RKind::DST10 => ffi::FFTW_RODFT10,
            R2RKind::DST11 => ffi::FFTW_RODFT11,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum C2CDirection {
    FORWARD,
    BACKWARD,
}

impl Into<i32> for C2CDirection {
    fn into(self) -> i32 {
        match self {
            C2CDirection::FORWARD => ffi::FFTW_FORWARD,
            C2CDirection::BACKWARD => ffi::FFTW_BACKWARD,
        }
    }
}

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
