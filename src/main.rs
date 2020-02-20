use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt::{Debug, Formatter, Error};
use std::ops::Mul;

mod vectors;

use vectors::{RowVec, ColVec};

struct SRow {
    elements: Vec<Cell>
}

impl SRow {
    fn new() -> SRow {
        SRow { elements: vec![] }
    }
    
    fn len(&self) -> usize {
        self.elements.len()
    }
}

struct Cell {
    column: usize,
    value: f64,
}

impl Cell {
    fn new(column: usize, value: f64) -> Cell {
        Cell { column, value }
    }
}

struct SMat {
    m: usize,
    n: usize,
    rows: Vec<SRow>,
}

impl SMat {
    fn new(m: usize, n: usize) -> SMat {
        let mut rows = Vec::with_capacity(m);
        for _ in 0..m {
            rows.push(SRow::new())
        }
        SMat { m, n, rows }
    }

    fn from_path(path: &str) -> SMat {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        reader.read_line(&mut line).unwrap();

        let values = line.split(" ").collect::<Vec<&str>>();

        let m = values[1].parse::<usize>().unwrap();
        let n = values[3].trim().parse::<usize>().unwrap();

        let mut m = SMat::new(m, n);

        for l in reader.lines() {
            let line = l.unwrap();
            let substring = &line[4..];
            let comma_idx = substring.find(':').unwrap();
            let row_id = substring[0..comma_idx].parse::<usize>().unwrap();
            let values = substring[comma_idx + 2..].split(' ')
                .map(|it| it.parse::<i32>().unwrap())
                .collect::<Vec<i32>>();

            while m.rows.len() <= row_id {
                m.rows.push(SRow::new())
            }

            for value in values {
                if value != -1 {
                    m.rows[row_id].elements.push(Cell::new(value as usize, 1.0))
                }
            }
        }

        m
    }
}

impl Debug for SMat {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "SparseMatrix: {} by {}\n", self.m, self.n).unwrap();
        for i in 0..self.rows.len() {
            let row = &self.rows[i];
            write!(f, "row {}: ", i).unwrap();
            for cell in &row.elements {
                write!(f, "{}", cell.column).unwrap();
                if cell.value != 0.0 {
                    write!(f, ":{}", cell.value).unwrap();
                }
                write!(f, " ").unwrap();
            }
            if i != self.rows.len() - 1 {
                write!(f, "-1\n").unwrap();
            } else {
                write!(f, "-1").unwrap();
            }
        };
        Result::Ok(())
    }
}

impl<'a> Mul<&'a SMat> for &'a vectors::RowVec {
    type Output = vectors::RowVec;

    fn mul(self, rhs: &SMat) -> Self::Output {
        assert_eq!(self.len(), rhs.n);
        let mut res = RowVec::zeroes(rhs.n);
        for i in 0..rhs.m {
            for cell in &rhs.rows[i].elements {
                res[cell.column] += self[i] * cell.value;
            }
        }
        res
    }
}

impl<'a> Mul<&'a vectors::ColVec> for &'a vectors::RowVec {
    type Output = f64;

    fn mul(self, rhs: &ColVec) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self[i] * rhs[i];
        }
        sum
    }
}

fn m_to_h(m: &mut SMat) {
    for i in 0..m.m {
        let nnz = m.rows[i].len();
        for cell in &mut m.rows[i].elements {
            cell.value /= nnz as f64;
        }
    }
}

fn absorbent_node_vec(m: &SMat) -> ColVec {
    let mut result = ColVec::zeroes(m.m);
    for i in 0..m.m {
        if m.rows[i].len() == 0 {
            result[i] = 1.0;
        }
    }
    result
}

fn main() {
    let nb_steps = 10;
    let alpha = 0.99;

    let mut matrix = SMat::from_path("exemple.dat");
    m_to_h(&mut matrix);

    /**************************/
    /* Solution to Exercise 3 */
    /**************************/
    let mut r_t = RowVec::init_with(matrix.m, |_| 1.0 / matrix.n as f64);
    for _ in 0..nb_steps {
        r_t = &r_t * &matrix;
    }
    println!("{:?}", r_t);

    /**************************/
    /* Solution to Exercise 5 */
    /**************************/
    let mut r_t = RowVec::init_with(matrix.m, |_| 1.0 / matrix.n as f64);
    let abs_nodes = absorbent_node_vec(&matrix);
    let abs_nodes_over_n = &abs_nodes / matrix.n as f64;
    for _ in 0..nb_steps {
        let sum = &r_t * &abs_nodes_over_n;
        r_t = &r_t * &matrix;
        for i in 0..r_t.len() {
            r_t[i] += sum;
        }
    }
    println!("{:?}", r_t);

    /**************************/
    /* Solution to Exercise 8 */
    /**************************/
    let mut r_t = RowVec::init_with(matrix.m, |_| 1.0 / matrix.n as f64);
    let abs_nodes = absorbent_node_vec(&matrix);
    for _ in 0..nb_steps {
        let rhs = (alpha * (&r_t * &abs_nodes) + 1.0 - alpha) / matrix.n as f64;
        r_t *= alpha;
        r_t = &r_t * &matrix;
        for i in 0..r_t.len() {
            r_t[i] += rhs;
        }
    }
    println!("{:?}", r_t);
}