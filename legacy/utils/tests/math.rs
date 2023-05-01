#![feature(test)]
extern crate anna_utils;
extern crate rand;

#[test]
fn combinations_check() {
  use anna_utils::math::combinations;
  assert!(combinations(7, 5).len() == 21);
}

#[test]
fn sample_check() {
  use anna_utils::math::sample;

  let mut rng = rand::thread_rng();

  for _ in 1..20 {
    let i = sample(&mut rng, vec![0., 1.0, 0.0]);
    let j = sample(&mut rng, vec![0., 0.000000000000000000001, 0.]);
    assert!(i == 1);
    assert!(j == 1);
  }
}

#[test]
fn var_check() {
  use anna_utils::math::var;

  let x = var(&vec![-125.0, 101.5625, -23.4375, -46.875, -294.92188, -416.01563, -316.40625]);
  println!("{:?}", x);
}
