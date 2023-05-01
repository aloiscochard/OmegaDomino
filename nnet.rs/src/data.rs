extern crate rand;
extern crate serde;

use self::rand::Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, Tensor, TensorSystem};

fn sample<A: Clone, R: Rng>(
  sys: &mut TensorSystem,
  rng: &mut R,
  tensors: &Vec<Tensor<f32>>,
  values: &Vec<A>,
  amount_: usize,
) -> Result<(Vec<Tensor<f32>>, Vec<A>), Error> {
  use rand::seq::sample_indices;

  let amount = usize::min(values.len(), amount_);

  let is = sample_indices(rng, values.len(), amount);
  let is_i: Vec<i32> = is.iter().map(|&i| i as i32).collect();

  let mut ts = Vec::new();
  for t in tensors.iter() {
    ts.push(sys.gather(&t, &is_i)?);
  }

  let mut vs = Vec::new();
  for i in is {
    vs.push(values[i].clone());
  }

  Ok((ts, vs))
}

/** Circular Buffer
 **/
#[derive(Clone)]
pub struct CBuffer<A> {
  pub size: usize,
  pub tensors: Vec<Tensor<f32>>,
  pub values: Vec<A>,
  i: usize,
}

impl<A> CBuffer<A> {
  pub fn new(size: usize, tensors: Vec<Tensor<f32>>) -> CBuffer<A> {
    CBuffer { size: size, tensors: tensors, values: Vec::new(), i: 0 }
  }

  pub fn sample<R: Rng>(
    &self,
    sys: &mut TensorSystem,
    rng: &mut R,
    amount: usize,
  ) -> Result<(Vec<Tensor<f32>>, Vec<A>), Error>
  where
    A: Clone,
  {
    sample(sys, rng, &self.tensors, &self.values, amount)
  }

  pub fn push(
    &mut self,
    sys: &mut TensorSystem,
    xss: Vec<(Vec<Tensor<f32>>, A)>,
  ) -> Result<usize, Error> {
    let len = xss.len();

    for (xs, value) in xss {
      let size = self.values.len();
      if size == self.size {
        if self.i == self.size {
          self.i = 0;
        }
        self.values[self.i] = value;

        for t in 0..self.tensors.len() {
          self.tensors[t] = sys.row_set(&self.tensors[t], self.i as i32, &xs[t])?;
        }
      } else {
        self.values.push(value);

        for t in 0..self.tensors.len() {
          self.tensors[t] = sys.concat(&self.tensors[t], &xs[t])?;
        }
      }

      self.i += 1;
    }

    Ok(len)
  }

  pub fn push_values(&mut self, values: Vec<A>) -> Result<usize, Error> {
    let len = values.len();

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

    Ok(len)
  }

  pub fn usage(&self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }
}

impl<A> Serialize for CBuffer<A>
where
  A: Serialize,
{
  fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    use crate::TensorS;
    let tensors: Vec<TensorS<f32>> =
      self.tensors.iter().map(|t| TensorS { tensor: t.clone() }).collect();
    let values: Vec<&A> = self.values.iter().collect();
    (self.size, tensors, values, self.i).serialize(s)
  }
}

impl<'de, A> Deserialize<'de> for CBuffer<A>
where
  A: Deserialize<'de>,
{
  fn deserialize<D>(d: D) -> Result<CBuffer<A>, D::Error>
  where
    D: Deserializer<'de>,
  {
    use crate::TensorS;
    Deserialize::deserialize(d).and_then(
      |(size, tensors, values, i): (usize, Vec<TensorS<f32>>, Vec<A>, usize)| {
        Ok(CBuffer {
          size: size,
          tensors: tensors.iter().map(|ts| ts.tensor.clone()).collect(),
          values: values,
          i: i,
        })
      },
    )
  }
}

/** Reservoir Sampling (incremental), Algorithm R
 **  https://en.wikipedia.org/wiki/Reservoir_sampling
 **/

#[derive(Clone)]
pub struct RSampler<A> {
  pub size: usize,
  pub tensors: Vec<Tensor<f32>>,
  pub values: Vec<A>,
  p_min: f32,
  i: usize,
}

impl<A> RSampler<A> {
  pub fn new(size: usize, p_min: f32, tensors: Vec<Tensor<f32>>) -> RSampler<A> {
    RSampler { size: size, tensors: tensors, values: Vec::new(), p_min: p_min, i: 0 }
  }

  pub fn sample<R: Rng>(
    &self,
    sys: &mut TensorSystem,
    rng: &mut R,
    amount: usize,
  ) -> Result<(Vec<Tensor<f32>>, Vec<A>), Error>
  where
    A: Clone,
  {
    sample(sys, rng, &self.tensors, &self.values, amount)
  }

  pub fn push<R: Rng>(
    &mut self,
    sys: &mut TensorSystem,
    rng: &mut R,
    xss: Vec<(Vec<Tensor<f32>>, A)>,
  ) -> Result<usize, Error> {
    use self::rand::distributions::{Distribution, Range};

    let mut count = 0;

    for (xs, value) in xss {
      let size = self.values.len();
      if size == self.size {
        let p = (1.0 / self.i as f32) * (self.size as f32);
        let p_norm = if p < self.p_min { self.p_min } else { p };

        if rng.gen::<f32>() < p_norm {
          count += 1;

          let range = Range::new(0, self.size);
          let j = range.sample(rng);

          self.values[j] = value;

          for t in 0..self.tensors.len() {
            self.tensors[t] = sys.row_set(&self.tensors[t], j as i32, &xs[t])?;
          }
        }
      } else {
        count += 1;

        self.values.push(value);

        for t in 0..self.tensors.len() {
          self.tensors[t] = sys.concat(&self.tensors[t], &xs[t])?;
        }
      }
      self.i += 1;
    }

    Ok(count)
  }

  pub fn usage(&self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }
}

impl<A> Serialize for RSampler<A>
where
  A: Serialize,
{
  fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    use crate::TensorS;
    let tensors: Vec<TensorS<f32>> =
      self.tensors.iter().map(|t| TensorS { tensor: t.clone() }).collect();
    let values: Vec<&A> = self.values.iter().collect();
    (self.size, tensors, values, self.p_min, self.i).serialize(s)
  }
}

impl<'de, A> Deserialize<'de> for RSampler<A>
where
  A: Deserialize<'de>,
{
  fn deserialize<D>(d: D) -> Result<RSampler<A>, D::Error>
  where
    D: Deserializer<'de>,
  {
    use crate::TensorS;
    Deserialize::deserialize(d).and_then(
      |(size, tensors, values, p_min, i): (usize, Vec<TensorS<f32>>, Vec<A>, f32, usize)| {
        Ok(RSampler {
          size: size,
          tensors: tensors.iter().map(|ts| ts.tensor.clone()).collect(),
          values: values,
          p_min: p_min,
          i: i,
        })
      },
    )
  }
}

pub mod tests {

  pub fn check_data<D, P, S>(
    data: &mut D,
    xs: Vec<f32>,
    push: P,
    sample: S,
  ) -> Result<bool, crate::Error>
  where
    P: Fn(
      &mut crate::TensorSystem,
      &mut D,
      &mut rand::StdRng,
      Vec<(Vec<crate::Tensor<f32>>, f32)>,
    ) -> Result<usize, crate::Error>,
    S: Fn(
      &mut crate::TensorSystem,
      &mut D,
      &mut rand::StdRng,
      usize,
    ) -> Result<(Vec<crate::Tensor<f32>>, Vec<f32>), crate::Error>,
  {
    use rand::{SeedableRng, StdRng};

    use crate::{graph_load, Tensor, TensorSystem};

    let mut rng: StdRng = SeedableRng::from_seed([0; 32]);

    let sys_graph = graph_load("./graphs/system.pb").unwrap();
    let mut sys = TensorSystem::new(&sys_graph).unwrap();
    // let mut buffer = CBuffer::new(xs.len() / 2, vec![Tensor::new(&[0])]);

    let mut xss = Vec::new();

    let len = xs.len();

    if len < 4 {
      return Ok(true);
    }

    for x in xs {
      let tensor = Tensor::new(&[1]).with_values(&[x])?;
      xss.push((vec![tensor], x));
    }

    push(&mut sys, data, &mut rng, xss)?;

    let (vectors, values) = sample(&mut sys, data, &mut rng, len / 4)?;

    let vectors_xs: Vec<f32> = vectors[0].iter().map(|&i| i).collect();

    Ok(vectors_xs == values)
  }

  quickcheck! {
    fn cbuffer_sample(xs: Vec<f32>) -> bool {
      use crate::Tensor;
      use crate::data::CBuffer;

      let mut buffer = CBuffer::new(xs.len() / 2, vec![Tensor::new(&[0])]);

      check_data(&mut buffer, xs,
                 |mut sys, buffer, mut _rng, xss| {
                   buffer.push(&mut sys, xss)
                 },
                 |mut sys, buffer, mut rng, amount| {
                  buffer.sample(&mut sys, &mut rng, amount)
                 }).unwrap()
    }

    fn rsampler_sample(xs: Vec<f32>) -> bool {
      use crate::Tensor;
      use crate::data::RSampler;

      let mut sampler = RSampler::new(xs.len() / 2, 0.5, vec![Tensor::new(&[0])]);

      check_data(&mut sampler, xs,
                 |mut sys, sampler, mut rng, xss| {
                   sampler.push(&mut sys, &mut rng, xss)
                 },
                 |mut sys, sampler, mut rng, amount| {
                   sampler.sample(&mut sys, &mut rng, amount)
                 }).unwrap()
    }
  }
}
