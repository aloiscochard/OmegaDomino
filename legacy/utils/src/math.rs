extern crate rand;

use self::rand::Rng;
use std::collections::LinkedList;

#[derive(Clone)]
pub struct MovingAvg {
  size: usize,
  values: LinkedList<f32>,
}

impl MovingAvg {
  pub fn new(size: usize) -> MovingAvg {
    MovingAvg { values: LinkedList::new(), size: size }
  }

  pub fn is_full(&self) -> bool {
    self.values.len() == self.size
  }

  pub fn last(&self) -> Option<f32> {
    self.values.back().cloned()
  }

  pub fn len(&self) -> usize {
    self.values.len()
  }

  pub fn mean(&self) -> f32 {
    let r = 1.0 / self.values.len() as f32;
    let mut m = 0.0;
    for x in self.values.iter() {
      m += r * x;
    }
    m
  }

  pub fn push(&mut self, x: f32) -> () {
    self.values.push_back(x);
    if self.values.len() > self.size {
      self.values.pop_front();
    }
  }

  pub fn reset(&mut self) -> () {
    self.values.clear();
  }

  pub fn truncate(&self, size: usize) -> MovingAvg {
    if size > self.values.len() {
      self.clone()
    } else {
      MovingAvg {
        size: size,
        values: self.values.iter().skip(self.values.len() - size).map(|&x| x).collect(),
      }
    }
  }

  pub fn var(&self) -> f32 {
    let m = self.mean();

    let mut v = 0.0;

    for x in self.values.iter() {
      let xv = x - m;
      v += xv * xv;
    }

    v / (self.len() - 1) as f32
  }
}

pub fn mean(xs: &Vec<f32>) -> f32 {
  let r = 1.0 / xs.len() as f32;
  let mut m = 0.0;
  for x in xs {
    m += r * x;
  }
  m
}

/* Generate combinations with a fixed size length without repeating elements
 * or duplicate sequences of ordered elements. */
pub fn combinations(n: usize, length: usize) -> Vec<Vec<usize>> {
  let mut results: Vec<Vec<usize>> = Vec::new();

  for i in 0..(n - length) + 1 {
    let mut values: Vec<usize> = Vec::new();

    for j in 0..length {
      values.push(i + j);
    }

    results.push(values.clone());

    let mut stack: Vec<(usize, Vec<usize>)> = Vec::new();
    stack.push((length, values.clone()));

    while !stack.is_empty() {
      let (cursor_max, xs) = stack.remove(0);
      values = xs;

      // Find the highest incrementable digit, from right to left.
      let mut cursor: usize = 0;
      for j in (1..(cursor_max + 1)).rev() {
        for value in values.get(j as usize) {
          if *value < n {
            if (j + 1) == length || *values.get(j as usize + 1).unwrap_or(&0) > (value + 1) {
              cursor = j;
              break;
            }
          }
        }
      }

      if cursor != 0 {
        // Enumerate all incremented values for that digit.
        let n_max = *values.get(cursor_max + 1).unwrap_or(&n);
        let ks = (*values.get(cursor).unwrap_or(&n) + 1)..n_max;
        for k in ks {
          values.push(k);
          values.swap_remove(cursor);
          if cursor >= 1 {
            stack.push(((cursor - 1), values.clone()));
          }
          results.push(values.clone());
        }
      }
    }
  }

  results
}

pub fn sample<R: Rng>(rng: &mut R, ps: Vec<f32>) -> usize {
  use self::rand::distributions::{Distribution, Weighted, WeightedChoice};

  let max = u32::max_value() - (u32::max_value() / 1024); // prevent rounding overflow when `as f32`

  let sum: f32 = ps.iter().sum();
  let r: f32 = max as f32 / sum;

  let mut xs: Vec<Weighted<usize>> = ps
    .iter()
    .enumerate()
    .filter_map(
      |(i, &p)| {
        if p != 0. {
          Some(Weighted { weight: (r * p) as u32, item: i })
        } else {
          None
        }
      },
    )
    .collect();

  let mut wc = WeightedChoice::new(&mut xs);
  wc.sample(rng)
}

pub fn sample_all<T, I, R: Rng>(rng: &mut R, iterable: I, amount: usize) -> Vec<T>
where
  I: IntoIterator<Item = T>,
  R: Rng,
{
  use self::rand::seq::sample_iter;

  match sample_iter(rng, iterable, amount) {
    Ok(xs) => xs,
    Err(xs) => {
      let mut ys = xs;
      rng.shuffle(&mut ys);
      ys
    }
  }
}

pub fn var(xs: &Vec<f32>) -> f32 {
  let m = mean(xs);

  let mut v = 0.0;

  for x in xs {
    let xv = x - m;
    v += xv * xv;
  }

  v / (xs.len() - 1) as f32
}
