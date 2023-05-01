#![feature(test)]
extern crate arrayfire;
extern crate neutrinic;
extern crate rand;

extern crate anna_learning;

use neutrinic::Array;

#[test]
fn cache_buffer() {
  use anna_learning::cache::Buffer;
  use arrayfire::{print, Dim4};

  let mut buffer = Buffer::new(3);

  buffer.push(&vec![(
    Array::new(&vec![0.1, 1.0], Dim4::new(&[2, 1, 1, 1])),
    Array::new(&vec![0.01, 10.0], Dim4::new(&[2, 1, 1, 1])),
  )]);

  print(&buffer.inputs);
  print(&buffer.outputs);

  buffer.push(&vec![(
    Array::new(&vec![0.2, 2.0], Dim4::new(&[2, 1, 1, 1])),
    Array::new(&vec![0.02, 20.0], Dim4::new(&[2, 1, 1, 1])),
  )]);

  print(&buffer.inputs);
  print(&buffer.outputs);

  buffer.push(&vec![(
    Array::new(&vec![0.3, 3.0], Dim4::new(&[2, 1, 1, 1])),
    Array::new(&vec![0.03, 30.0], Dim4::new(&[2, 1, 1, 1])),
  )]);

  print(&buffer.inputs);
  print(&buffer.outputs);

  buffer.push(&vec![(
    Array::new(&vec![0.4, 4.0], Dim4::new(&[2, 1, 1, 1])),
    Array::new(&vec![0.04, 40.0], Dim4::new(&[2, 1, 1, 1])),
  )]);

  print(&buffer.inputs);
  print(&buffer.outputs);

  assert!(true == true);
}

#[test]
fn cache_reservoir() {
  use anna_learning::cache::Reservoir;
  use arrayfire::{print, Dim4};
  use rand::{SeedableRng, StdRng};

  let seed: &[_] = &[0];
  let mut rng: StdRng = SeedableRng::from_seed(seed);

  let mut buffer = Reservoir::new(16);

  for i_ in 1..36000 {
    let i = i_ as f32;
    buffer.push(
      &mut rng,
      &vec![(
        Array::new(&vec![i * 0.1, i * 1.0], Dim4::new(&[2, 1, 1, 1])),
        Array::new(&vec![i * 0.001, i * 100.0], Dim4::new(&[2, 1, 1, 1])),
      )],
    );
  }

  print(&buffer.inputs);
  print(&buffer.outputs);
}
