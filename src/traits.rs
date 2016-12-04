
use num_complex::Complex;
use ffi;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

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
