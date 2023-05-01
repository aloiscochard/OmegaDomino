#[macro_use]
extern crate log;
extern crate nnet;
extern crate rand;
extern crate rayon;

extern crate anna_learning;
extern crate anna_model;
extern crate anna_simulation;
extern crate anna_utils;

fn main() -> () {
  use std::{collections::HashSet, env::args, iter::FromIterator, ops::Neg, path::Path};

  use nnet::Network;
  use rayon::{current_num_threads, prelude::*};

  use anna_model::Money;

  use anna_learning::{
    agent_episode::{Episode, Policy},
    qnet::QNet,
    snet::SNet,
    snet_lex::SNetLex,
  };

  use anna_learning::plan::*;

  use anna_simulation::{
    perf::run_benchmark,
    players::*,
    players_kuhn::Kuhn2,
    players_lex::Lex,
    players_mul::{PlayersMul2, PlayersMulX},
    Sim,
  };

  use anna_utils::{bincode, logging, math, random};

  logging::init();

  let threads = current_num_threads();

  assert!(threads >= 2);

  let fund_init = Money::new(100, 0);

  let path_data_str = args().nth(1).unwrap();
  let ref path_data = Path::new(path_data_str.as_str());
  let ref path_graphs = path_data.join("graphs");
  let ref path_networks = path_data.join("networks");
  let ref path_synthetic = path_data.join("synthetic");

  // let plan = plan_kuhn2();
  let plan_b = plan_texas_limit_n(2, &path_synthetic);
  let plan = plan_texas_limit_zero_n(2, &path_synthetic);

  // let network_a_path = Path::new("../resources/networks/qnet-texas_limit-6-28-[1024].data");
  let network_a_path = Path::new("../resources/training/archives.20190418/tl-2z/network-1-q.data");

  // let network_b_path = Path::new("/tmp/training/network-b.data");

  let network_a: Network = bincode::deserialize_from_file(network_a_path).unwrap();
  // let network_b: Network = bincode::deserialize_from_file(network_b_path).unwrap();

  let ref mut thread_rng = rand::thread_rng();

  let ref sim = plan.sim();

  let snet = SNet::new(path_graphs, &plan.profile, &plan.eval, &plan.snet_hiddens).unwrap();

  let cards_encoder_create = |rng| match plan.cards_encoders(rng, &snet, &plan.eval, path_networks)
  {
    CardsEncoders::Binary(encoder) => encoder,
    // CardsEncoders::StrengthNNet(encoder) => encoder,
    _ => unimplemented!(),
  };

  let cards_encoder_strength_create =
    |rng| match plan_b.cards_encoders(rng, &snet, &plan_b.eval, path_networks) {
      CardsEncoders::StrengthNNet(encoder) => encoder,
      _ => unimplemented!(),
    };
  let actions_encoder = plan.actions_encoder();

  let cards_encoder = cards_encoder_create(random::rseed(thread_rng));

  let qnet =
    QNet::new(path_graphs, sim, actions_encoder.as_ref(), &cards_encoder, &plan.qnet_hiddens)
      .unwrap();

  let mut benchmark_players = Vec::new();
  for thread_id in 0..threads {
    /*
    let (network_0, network_1) =
      if thread_id % 2 == 0 { (&network_a, &network_b) } else { (&network_b, &network_a) };

    let mut xs = Vec::new();
    for player_id in 0..(plan.profile.players - 1) {
      let mut rng = random::rseed(thread_rng);
      let mut episode = Episode::new(&mut rng,
                                     actions_encoder.as_ref(),
                                     cards_encoder_create(random::rseed(thread_rng)),
                                     qnet.p_predict().unwrap(),
                                     qnet.q_predict().unwrap(),
                                     &qnet).unwrap();

      episode.reset(network_0, Policy::P).unwrap();

      xs.push((episode, HashSet::from_iter(vec![player_id])));
    }

    let players0 = PlayersMulX(xs);
    let players0_mask = players0.mask();

    let players1 = {
      let mut rng = random::rseed(thread_rng);
      let mut episode = Episode::new(&mut rng,
                                     actions_encoder.as_ref(),
                                     cards_encoder_create(random::rseed(thread_rng)),
                                     qnet.p_predict().unwrap(),
                                     qnet.q_predict().unwrap(),
                                     &qnet).unwrap();

      episode.reset(network_1, Policy::P).unwrap();
      episode
    };
    */

    let mut rng = random::rseed(thread_rng);
    let mut players0 = Episode::new(
      &mut rng,
      actions_encoder.as_ref(),
      cards_encoder_create(random::rseed(thread_rng)),
      qnet.p_predict().unwrap(),
      qnet.q_predict().unwrap(),
      &qnet,
    )
    .unwrap();

    players0.reset(&network_a, Policy::P).unwrap();

    let mut xs = Vec::new();
    for player_id in 1..plan.profile.players {
      let players_lex = SNetLex::new(
        random::rseed(thread_rng),
        cards_encoder_strength_create(random::rseed(thread_rng)),
        &plan.action_class,
        &plan.profile,
      );
      xs.push((players_lex, HashSet::from_iter(vec![player_id])));
    }

    let players1 = PlayersMulX(xs);
    let players1_mask = players1.mask();

    let players = PlayersMul2((players0, HashSet::from_iter(vec![0])), (players1, players1_mask));

    benchmark_players.push(players);
  }

  let batch_size = 128;

  let mut totals: Vec<f32> = vec![0.0; 2];
  let mut session_id = 1;

  loop {
    let results: Vec<(usize, f32)> = benchmark_players
      .par_iter_mut()
      .enumerate()
      .map(|(thread_id, players)| {
        let funds = (0..plan.profile.players).map(|seat_id| (seat_id, fund_init)).collect();
        let rates = run_benchmark(sim, &plan.eval, &funds, batch_size / threads, players);

        // let network_id = if thread_id % 2 == 0 { 0 } else { 1 };
        let network_id = 0;

        (network_id, rates[plan.profile.players - 1].neg())
      })
      .collect();

    // let mut network_rates = vec![Vec::new(); 2];
    let mut network_rates = vec![Vec::new(); 1];

    for (i, rate) in results {
      network_rates[i].push(rate);
    }

    let mut bench_summary = String::new();

    for (i, rates) in network_rates.iter().enumerate() {
      let session = math::mean(rates);
      totals[i] += session;
      let total = totals[i] / session_id as f32;
      bench_summary.push_str(format!(" {:08.0} ({:08.0})", total, session).as_str());
    }

    // Logging
    println!(" {:12} | {}", session_id * batch_size, bench_summary);

    session_id = session_id + 1;
  }
}
