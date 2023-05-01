#![feature(test)]
extern crate rand;

extern crate anna_eval;
extern crate anna_model;
extern crate anna_simulation;

#[test]
fn kuhn_sample_game() {
  use rand::{SeedableRng, StdRng};

  use anna_eval::Eval;
  use anna_model::{classifiers::ActionKuhn, profile::profile_kuhn, Money};
  use anna_simulation::{engine::table_game_simulate, players::PlayersRand, Sim};

  let seed: &[_] = &[0];

  let mut rng: StdRng = SeedableRng::from_seed(seed);

  let ref eval = Eval::naive();

  let blind = Money::new(1, 0);
  let fund = Money::new(100, 0);

  let funds = vec![
    (0, fund),
    (1, fund),
    /* (2, fund), */
  ];

  let ref action_kuhn = ActionKuhn {};

  let ref sim = Sim { action_class: action_kuhn, blind_biggest: blind, profile: profile_kuhn(3) };

  let ref mut players = PlayersRand { action_class: action_kuhn, blind_biggest: blind, rng: rng };

  for _ in 0..10 {
    let (_, score) =
      table_game_simulate(&mut rng, sim, eval, players, &funds, 0).map_err(|(_, err)| err).expect(
        "table simulation \
                                                                           failed.",
      );

    //println!("{:?}", log.rounds);
    println!("{:?}", score);
  }

  assert!(true == true);
}

#[test]
fn leduc_sample_game() {
  use rand::{SeedableRng, StdRng};

  use anna_eval::Eval;
  use anna_model::{classifiers::ActionLimit, profile::profile_leduc, Money};
  use anna_simulation::{engine::table_game_simulate, players::PlayersRand, Sim};

  let seed: &[_] = &[0];

  let mut rng: StdRng = SeedableRng::from_seed(seed);

  let ref eval = Eval::naive();

  let blind = Money::new(1, 0);
  let fund = Money::new(100, 0);

  let funds = vec![(0, fund), (1, fund)];
  //(2, fund),

  let ref profile = profile_leduc(3);

  let ref action_leduc = ActionLimit { raises: profile.clone().limit.unwrap().raises };

  let ref sim = Sim { action_class: action_leduc, blind_biggest: blind, profile: profile.clone() };

  let ref mut players = PlayersRand { action_class: action_leduc, blind_biggest: blind, rng: rng };

  for _ in 0..10 {
    let (_, score) =
      table_game_simulate(&mut rng, sim, eval, players, &funds, 0).map_err(|(_, err)| err).expect(
        "table simulation \
                                                                           failed.",
      );

    //println!("{:?}", log.rounds);
    println!("{:?}", score);
  }

  assert!(true == true);
}
