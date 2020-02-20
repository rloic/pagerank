use std::ops::{Index, IndexMut, Mul, Div, MulAssign};
use std::fmt::{Debug, Formatter, Error};

pub trait Transpose<T> {
    fn transpose(&self) -> T;
}

pub struct ColVec {
    values: Vec<f64>
}

impl ColVec {
    pub fn zeroes(len: usize) -> ColVec {
        ColVec { values: vec![0.0; len] }
    }

    pub fn init_with<InitFn>(len: usize, init: InitFn) -> ColVec
        where InitFn: Fn(usize) -> f64
    {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            values.push(init(i));
        };
        ColVec { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

impl Transpose<RowVec> for ColVec {
    fn transpose(&self) -> RowVec {
        RowVec { values: self.values.clone() }
    }
}

impl Index<usize> for ColVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for ColVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Debug for ColVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if self.values.len() != 0 {
            write!(f, "{:?}", self.values[0]).unwrap();
        }
        for i in 1..self.values.len() {
            write!(f, "\n{:?}", self.values[i]).unwrap();
        }
        Result::Ok(())
    }
}

impl<'a> Mul<f64> for &'a ColVec {
    type Output = ColVec;

    fn mul(self, rhs: f64) -> Self::Output {
        ColVec::init_with(self.len(), |idx| self[idx] * rhs)
    }
}

impl<'a> Div<f64> for &'a ColVec {
    type Output = ColVec;

    fn div(self, rhs: f64) -> Self::Output {
        ColVec::init_with(self.len(), |idx| self[idx] / rhs)
    }
}

pub struct RowVec {
    values: Vec<f64>
}

impl RowVec {
    pub fn zeroes(len: usize) -> RowVec {
        RowVec { values: vec![0.0; len] }
    }

    pub fn init_with<InitFn>(len: usize, init: InitFn) -> RowVec
        where InitFn: Fn(usize) -> f64
    {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            values.push(init(i));
        };
        RowVec { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

impl Transpose<ColVec> for RowVec {
    fn transpose(&self) -> ColVec {
        ColVec { values: self.values.clone() }
    }
}

impl Index<usize> for RowVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for RowVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Debug for RowVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", "[").unwrap();
        if self.values.len() != 0 {
            write!(f, "{:?}", self.values[0]).unwrap();
        }
        for i in 1..self.values.len() {
            write!(f, ", {:?}", self.values[i]).unwrap();
        }
        write!(f, "{}", "]").unwrap();
        Result::Ok(())
    }
}

impl<'a> Mul<f64> for &'a RowVec {
    type Output = RowVec;

    fn mul(self, rhs: f64) -> Self::Output {
        RowVec::init_with(self.len(), |idx| self[idx] * rhs)
    }
}

impl MulAssign<f64> for RowVec {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.len() {
            self[i] *= rhs;
        }
    }
}

impl<'a> Div<f64> for &'a RowVec {
    type Output = RowVec;

    fn div(self, rhs: f64) -> Self::Output {
        RowVec::init_with(self.len(), |idx| self[idx] / rhs)
    }
}