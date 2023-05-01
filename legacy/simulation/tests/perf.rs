#![feature(test)]
extern crate env_logger;
extern crate log;
extern crate rand;
extern crate test;

extern crate anna_eval;
extern crate anna_model;
extern crate anna_simulation;

use test::Bencher;

#[bench]
fn kuhn2_bench(bencher: &mut Bencher) {
  use env_logger;
  env_logger::init().unwrap();

  use rand::{SeedableRng, StdRng};
  use std::{collections::HashSet, iter::FromIterator};

  use anna_eval::Eval;
  use anna_model::{classifiers::ActionKuhn, profile::profile_kuhn, Money};
  use anna_simulation::{
    perf::run_benchmark,
    players::{PlayersFold, PlayersRand},
    players_kuhn::Kuhn2,
    players_mul::PlayersMulX,
    Sim,
  };

  let seed: &[_] = &[0];
  let rng: StdRng = SeedableRng::from_seed(seed);

  let ref eval = Eval::naive();

  let blind = Money::new(1, 0);
  let fund = Money::new(100, 0);

  let funds = vec![
    (0, fund),
    (1, fund),
    /* (2, fund), */
  ];

  let ref action_kuhn = ActionKuhn {};

  let ref sim = Sim { action_class: action_kuhn, blind_biggest: blind, profile: profile_kuhn(2) };

  let ref mut players_fold = PlayersFold::new(action_kuhn);

  let ref mut players_rand =
    PlayersRand { action_class: action_kuhn, blind_biggest: blind, rng: rng };

  bencher.iter(|| {
    let rng: StdRng = SeedableRng::from_seed(seed);
    let players0 = Kuhn2::new(rng, action_kuhn);
    let rng: StdRng = SeedableRng::from_seed(seed);
    let players1 = Kuhn2::new(rng, action_kuhn);

    let players = PlayersMulX(vec![
      (players0, HashSet::from_iter(vec![0])),
      (players1, HashSet::from_iter(vec![1])),
    ]);

    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let rates = run_benchmark(sim, eval, players, &funds, 10_000);
    println!("{:?}", rates);
  });

  assert!(true == true);
}
