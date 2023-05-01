extern crate ctrlc;
extern crate nnet;
extern crate rand;

extern crate anna_eval;
extern crate anna_learning;
extern crate anna_model;

use anna_eval::Eval;
use anna_model::{cards::Card, profile::Profile};

/**
 * snet_train
 *
 * cd learning
 * cargo run --release --bin snet-train ../resources/
 *
 **/

type Data = Vec<(Vec<Vec<Card>>, f32)>;

fn sample_hands_full(eval: &Eval, profile: &Profile, size: usize) -> Data {
  use rand::seq::sample_indices;

  let ref mut rng = rand::thread_rng();

  let mut xs = Vec::new();

  let cards_len = profile.rounds.iter().sum::<usize>();
  let rounds_len = profile.rounds.len();

  for _ in 0..size {
    let mut cards = Vec::new();

    let mut cards_idx = sample_indices(rng, profile.deck.len(), cards_len);

    for round_id in 0..rounds_len {
      let mut xs = Vec::new();
      for _ in 0..profile.rounds[round_id] {
        let card_id = cards_idx.remove(0);
        xs.push(profile.deck[card_id]);
      }
      cards.push(xs);
    }
    let hand: Vec<Card> = cards.iter().flatten().cloned().collect();
    let score = eval.score(&hand);

    xs.push((cards, score));
  }

  xs
}

fn sample_hands(eval: &Eval, profile: &Profile, accuracy: usize, size: usize) -> Data {
  use anna_eval::strength::strength_;
  use rand::{
    distributions::{Distribution, Range},
    seq::sample_indices,
  };

  let ref mut rng = rand::thread_rng();

  let mut xs = Vec::new();

  let rounds_len = profile.rounds.len();
  let rounds_range = Range::new(1, rounds_len + 1);

  for _ in 0..size {
    let rounds = rounds_range.sample(rng);
    let mut cards = Vec::new();

    let mut cards_idx =
      sample_indices(rng, profile.deck.len(), profile.rounds.iter().take(rounds).sum::<usize>());

    for round_id in 0..rounds {
      let mut xs = Vec::new();
      for _ in 0..profile.rounds[round_id] {
        let card_id = cards_idx.remove(0);
        xs.push(profile.deck[card_id]);
      }
      cards.push(xs);
    }

    let score = strength_(
      rng,
      eval,
      &profile.rounds,
      profile.players,
      &profile.deck,
      accuracy,
      &cards[0],
      &cards[1..].iter().flatten().cloned().collect(),
    );

    xs.push((cards, score));
  }

  xs
}

fn main() -> () {
  use anna_learning::{plan::*, snet::SNet};
  use nnet::tensor_of;
  use std::{
    env::args,
    path::Path,
    sync::{
      atomic::{AtomicBool, Ordering},
      mpsc::{self, Receiver, Sender},
      Arc,
    },
    thread,
  };

  let dropout_rate = 0.5;

  let path_data_str = args().nth(1).unwrap();
  let ref path_data = Path::new(path_data_str.as_str());
  let ref path_graphs = path_data.join("graphs");
  let ref path_synthetic = path_data.join("synthetic");

  // let plan = plan_kuhn2();
  // let plan = plan_kuhn3();
  // let plan = plan_leduc2();
  // let plan = plan_leduc_french2();
  // let plan = plan_leduc_french3();
  // let plan = plan_leduc_french6();
  // let plan = plan_cochard2();

  let plan = plan_texas_limit_n(6, path_synthetic);

  let ref profile = plan.profile;
  let ref eval = plan.eval;

  let validation_size = plan.snet_params.batch_size * 64;

  let snet = SNet::new(path_graphs, profile, eval, &plan.snet_hiddens).unwrap();

  let (tx, rx): (Sender<Data>, Receiver<Data>) = mpsc::channel();

  let running = Arc::new(AtomicBool::new(true));

  let worker = {
    let r = running.clone();
    let plan = plan.clone();
    thread::spawn(move || {
      while r.load(Ordering::SeqCst) {
        tx.send(sample_hands(
          &plan.eval,
          &plan.profile,
          plan.snet_params.accuracy,
          plan.snet_params.batch_size,
        ))
        .unwrap();
      }
    })
  };

  let mut train = snet.train().unwrap();

  println!("training ...");

  let r = running.clone();

  ctrlc::set_handler(move || {
    r.store(false, Ordering::SeqCst);
  })
  .unwrap();

  // Monte carlo training
  let mut epoch = 0;
  while running.load(Ordering::SeqCst) {
    let data = rx.recv().unwrap();
    let size = data.len();

    let mut is = Vec::new();
    let mut os = Vec::new();

    for (cards, score) in data {
      is.extend(&snet.to_vector(&profile, &cards));
      os.push(score);
    }

    let input = tensor_of(&[size as u64, snet.inputs as u64], &is).unwrap();
    let target = tensor_of(&[size as u64, 1], &os).unwrap();

    let accuracy = SNet::run(&mut train, 0.0, 0.0, &input, &target).unwrap();
    let cost =
      SNet::run(&mut train, plan.snet_params.learning_rate, dropout_rate, &input, &target).unwrap();

    println!("{:7}: {:.4}\t {:.4}", epoch, accuracy, cost);
    epoch += 1;
  }

  // Save network
  println!("saving ...");
  let network = train.context.get().unwrap();
  snet.network_save(&plan.profile, &Path::new("./"), &network).unwrap();

  // Compare accuracy compared to eval.
  println!("validation ...");
  let xs = sample_hands_full(&plan.eval, &plan.profile, validation_size);
  let ys = {
    let mut is = Vec::new();

    for (cards, score) in xs.iter() {
      is.extend(&snet.to_vector(&profile, &cards));
    }

    let input = tensor_of(&[validation_size as u64, snet.inputs as u64], &is).unwrap();
    train.predict(&input).unwrap()
  };

  let mut errors = 0;

  for i in 1..xs.len() {
    let x_a = &xs[i - 1];
    let x_b = &xs[i];

    let y_a = &ys[i - 1];
    let y_b = &ys[i];

    if (y_a > y_b && x_a < x_b) || (y_a < y_b && x_a > x_b) {
      errors += 1;
    }
  }

  println!("eval accuracy: {:.4}", errors as f32 / validation_size as f32);

  worker.join().unwrap();
}
