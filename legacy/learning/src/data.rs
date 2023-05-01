extern crate rand;

use self::rand::Rng;

pub fn sample_indices<R: Rng>(rng: &mut R, len: usize, amount_: usize) -> Vec<usize> {
  use rand::seq::sample_indices;
  let amount = usize::min(len, amount_);
  sample_indices(rng, len, amount)
}

/** Circular Buffer
 **/
#[derive(Clone)]
pub struct CBuffer<A> {
  pub size: usize,
  pub values: Vec<A>,
  i: usize,
}

impl<A> CBuffer<A> {
  pub fn new(size: usize) -> CBuffer<A> {
    CBuffer { size: size, values: Vec::with_capacity(size), i: 0 }
  }

  pub fn get(&self, i: usize) -> &A {
    &self.values[i]
  }

  pub fn push(&mut self, values: Vec<A>) -> () {
    for value in values {
      let size = self.values.len();
      if size == self.size {
        if self.i == self.size {
          self.i = 0;
        }
        self.values[self.i] = value;
      } else {
        self.values.push(value);
      }

      self.i += 1;
    }
  }

  pub fn usage(&self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }

  pub fn len(&self) -> usize {
    self.values.len()
  }
}

/** Reservoir Sampling (incremental), Algorithm R
 **  https://en.wikipedia.org/wiki/Reservoir_sampling
 **/

#[derive(Clone)]
pub struct RSampler<A> {
  pub size: usize,
  values: Vec<A>,
  p_min: f32,
  i: usize,
}

impl<A> RSampler<A> {
  pub fn new(size: usize, p_min: f32) -> RSampler<A> {
    RSampler { size: size, values: Vec::with_capacity(size), p_min: p_min, i: 0 }
  }

  pub fn get(&self, i: usize) -> &A {
    &self.values[i]
  }

  pub fn push<R: Rng>(&mut self, rng: &mut R, values: Vec<A>) -> usize {
    use self::rand::distributions::{Distribution, Range};

    let mut count = 0;

    for value in values {
      let size = self.values.len();
      if size == self.size {
        let p = (1.0 / self.i as f32) * (self.size as f32);
        let p_norm = if p < self.p_min { self.p_min } else { p };

        if rng.gen::<f32>() < p_norm {
          count += 1;

          let range = Range::new(0, self.size);
          let j = range.sample(rng);

          self.values[j] = value;
        }
      } else {
        count += 1;

        self.values.push(value);
      }
      self.i += 1;
    }

    count
  }

  pub fn usage(&self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }

  pub fn len(&self) -> usize {
    self.values.len()
  }
}
