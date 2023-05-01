extern crate rand;

use self::rand::{Rng, StdRng};

pub fn rseed<R: Rng>(rng: &mut R) -> StdRng {
  use self::rand::SeedableRng;
  let seed: [u8; 32] = rng.gen();
  StdRng::from_seed(seed)
}
