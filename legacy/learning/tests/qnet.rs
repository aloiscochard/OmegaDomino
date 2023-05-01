#![feature(test)]
extern crate arrayfire;
extern crate env_logger;
extern crate neutrinic;
extern crate rand;

extern crate anna_eval;
extern crate anna_learning;
extern crate anna_model;
extern crate anna_simulation;

#[test]
fn qstate_update() {
  use rand::{SeedableRng, StdRng};

  use anna_eval::Eval;
  use anna_learning::qnet::{QNet, QState};
  use anna_model::{classifiers::ActionLimit, profile::profile_leduc};
  use anna_simulation::{engine::table_game_simulate, players::PlayersRand, Sim};

  use anna_model::Money;

  // Create simulation model
  let ref profile = profile_leduc(2);
  let ref action_class = ActionLimit { raises: profile.clone().limit.unwrap().raises };

  let blind = Money::new(1, 0);

  let ref sim = Sim { action_class: action_class, blind_biggest: blind, profile: profile.clone() };

  // Synthetize games
  let seed: &[_] = &[0];
  let mut rng: StdRng = SeedableRng::from_seed(seed);
  let ref eval = Eval::naive();

  let funds = vec![(0, blind * 100), (1, blind * 100)];

  let ref mut players = PlayersRand { action_class: action_class, blind_biggest: blind, rng: rng };

  for _ in 0..10 {
    let (log, score) =
      table_game_simulate(&mut rng, sim, eval, players, &funds, 0).map_err(|(_, err)| err).expect(
        "table simulation \
                                                                           failed.",
      );

    //println!("{:?}", log.rounds);
    println!("{:?}", score);

    // Create QNet and states
    let ref qnet = QNet::new(sim, &vec![8]);

    let player = score.winners[0];
    let player_cards = log
      .players_init
      .iter()
      .filter_map(
        |&(seat_id, _, ref cards)| {
          if seat_id == player {
            Some(cards.clone())
          } else {
            None
          }
        },
      )
      .next()
      .unwrap();

    let mut state = QState::new(qnet, player, &player_cards);

    for ev in log.events.iter() {
      state.update(qnet, ev);
      // println!("ev: {:?}", ev);
      // println!("{:?}", state);
    }

    println!("{:?}", state);
  }

  assert!(true == true);
}

#[test]
fn qstate_train() {
  use rand::{SeedableRng, StdRng};

  use anna_eval::Eval;
  use anna_model::{classifiers::ActionLimit, profile::profile_leduc, Money};
  use anna_simulation::{engine::table_game_simulate, players::PlayersRand, Sim};

  use anna_learning::{behavior, qnet::QNet};

  // Create simulation model
  let ref profile = profile_leduc(2);
  let ref action_class = ActionLimit { raises: profile.clone().limit.unwrap().raises };

  let blind = Money::new(1, 0);

  let ref sim = Sim { action_class: action_class, blind_biggest: blind, profile: profile.clone() };

  // Synthetize games
  let seed: &[_] = &[0];
  let mut rng: StdRng = SeedableRng::from_seed(seed);
  let ref eval = Eval::naive();

  let funds = vec![(0, blind * 100), (1, blind * 100)];

  let ref mut players = PlayersRand { action_class: action_class, blind_biggest: blind, rng: rng };

  // Create QNet and generate samples
  let ref qnet = QNet::new(sim, &vec![4]);
  let mut behaviors = Vec::new();

  for _ in 0..6 {
    let (log, _) =
      table_game_simulate(&mut rng, sim, eval, players, &funds, 0).map_err(|(_, err)| err).expect(
        "table simulation \
                                                                           failed.",
      );

    for (seat_id, _, ref player_cards) in log.players_init {
      behaviors.extend(behavior::extract_all(qnet, seat_id, &player_cards, &log.events));
    }
  }

  // Train network
  println!("{:?}", behaviors.len());

  use env_logger;
  use neutrinic::dataset::*;
  env_logger::init().unwrap();

  let training = behavior::training_set(qnet, behaviors);
  let validation = validation_set(training.clone(), eval_class());

  println!("training ...");
  use arrayfire::{RandomEngine, RandomEngineType::PHILOX_4X32_10};
  use neutrinic::{fnn::*, network_init, optimizer::*, recorder::*};

  let ref rnd = RandomEngine::new(PHILOX_4X32_10, Some(1));
  let ref mut network = network_init(&rnd, &qnet.p_schema);
  let ref mut optimizer = RMS::new(&qnet.p_schema, 0.1, 1.0, 0.95);
  let ref mut recorder = Recorder::void();
  train(rnd, &qnet.p_schema, network, optimizer, recorder, &training, &validation, 16, 4);

  assert!(true == true);
}
