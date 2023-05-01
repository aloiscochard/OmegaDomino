use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use rand::StdRng;

use nnet::Network;

use std::path::Path;
use std::{fs, io};

use anna_eval::Eval;
use anna_learning::agent_episode::Episode;
use anna_learning::plan::{CardsEncoders, Plan};
use anna_learning::qnet::QNet;
use anna_learning::{snet::SNet, snet_lex::SNetLex};
use anna_model::{
  encoders::{ActionsBinary, ActionsCompact, CardsEncoder},
  ActionClass,
};
use anna_utils::bincode;

pub enum Event {
  Network {
    epoch: usize,
    networks: Vec<(usize, (nnet::Network, nnet::Network))>,
    action_benchmark: bool,
    action_save: bool,
  },
  Abort,
}

fn networks_save(
  path_training: &Path,
  epoch: usize,
  networks: &[(usize, (Network, Network))],
) -> io::Result<()> where {
  let ref path_epoch = path_training.join(format!("epoch-{:012}/", epoch));
  fs::create_dir_all(path_epoch)?;

  // Saving
  for (i, (network_p, network_q)) in networks.iter() {
    let ref network_file = path_epoch.join(format!("network-{}-p.data", i));
    bincode::serialize_into_file(network_file, &network_p).unwrap();
    let ref network_file = path_epoch.join(format!("network-{}-q.data", i));
    bincode::serialize_into_file(network_file, &network_q).unwrap();
  }

  return Ok(());
}

pub fn worker_start<AC: ActionClass, E: Eval>(
  plan: Plan<AC, E>,
  threads: usize,
) -> (JoinHandle<()>, Sender<Event>)
where
  AC: Clone,
  E: Clone,
  AC: Send,
  E: Send,
  AC: 'static,
  E: 'static,
{
  use std::sync::mpsc;
  use std::thread;
  use std::{collections::HashSet, env::args, io::Write, iter::FromIterator, ops::Neg, path::Path};

  use rayon::prelude::*;

  use anna_learning::agent_episode::Policy;

  use anna_model::Money;

  use anna_simulation::{
    perf::run_benchmark,
    players::*,
    players_kuhn::Kuhn2,
    players_lex::Lex,
    players_mul::{PlayersMul2, PlayersMulX},
    Sim,
  };
  use nnet::Network;

  use anna_utils::{bincode, math, random};

  let (tx, rx): (Sender<Event>, Receiver<Event>) = mpsc::channel();

  (
    thread::spawn(move || {
      let epoch_end = args().nth(2).unwrap().parse::<usize>().unwrap();

      let path_data_str = args().nth(3).unwrap();
      let ref path_data = Path::new(path_data_str.as_str());
      let ref path_graphs = path_data.join("graphs");
      let ref path_networks = path_data.join("networks");

      let ref path_training_str = args().nth(4).unwrap();
      let ref path_training = Path::new(path_training_str.as_str());

      let ref mut thread_rng = rand::thread_rng();

      let fund_init = Money::new(100, 0);

      let ref benchmarking_log_path = path_training.join("benchmarking.log");

      let benchmarking_log_file = {
        use std::fs::OpenOptions;
        let mut options = OpenOptions::new();
        options.create(true).write(true).append(true);
        options.open(benchmarking_log_path).expect("Unable to open benchmarking log.")
      };

      let mut benchmarking_log = std::io::BufWriter::new(&benchmarking_log_file);

      let ref sim = plan.sim();

      let snet = SNet::new(path_graphs, &plan.profile, &plan.eval, &plan.snet_hiddens).unwrap();

      let cards_encoder_create = |rng| {
        match plan.cards_encoders(rng, &snet, &plan.eval, path_networks) {
          CardsEncoders::Binary(encoder) => encoder,
          // CardsEncoders::StrengthNNet(encoder) => encoder,
          _ => unimplemented!(),
        }
      };

      let plan_b = {
        use anna_learning::plan::*;
        let path_data_str = args().nth(3).unwrap();
        let ref path_data = Path::new(path_data_str.as_str());
        let ref path_synthetic = path_data.join("synthetic");
        plan_texas_limit_n(plan.profile.players, &path_synthetic)
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

      for _ in 0..threads {
        let players0 = {
          let mut rng = random::rseed(thread_rng);
          Episode::new(
            &mut rng,
            actions_encoder.as_ref(),
            cards_encoder_create(random::rseed(thread_rng)),
            qnet.p_predict().unwrap(),
            qnet.q_predict().unwrap(),
            &qnet,
          )
          .unwrap()
        };

        let mut xs = Vec::new();
        for player_id in 1..plan.profile.players {
          // let players_lex = Lex::new(random::rseed(thread_rng), 64, &plan.action_class, &plan.eval, &plan.profile);
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

        let players =
          PlayersMul2((players0, HashSet::from_iter(vec![0])), (players1, players1_mask));

        benchmark_players.push(players);
      }

      let mut rate_max = -1_000_000.0;

      loop {
        match rx.recv().unwrap() {
          Event::Abort => {
            break;
          }
          Event::Network { epoch, networks, action_benchmark, action_save } => {
            // Saving
            if action_save {
              networks_save(path_training, epoch, &networks).unwrap();
            }

            // Benchmark
            if action_benchmark {
              let results: Vec<Vec<(usize, f32)>> = benchmark_players
                .par_iter_mut()
                .map(|players| {
                  networks
                    .iter()
                    .map(|(i, (_, network))| {
                      let funds =
                        (0..plan.profile.players).map(|seat_id| (seat_id, fund_init)).collect();

                      ((players.0).0).reset(network, Policy::P).unwrap();

                      let rates = run_benchmark(sim, &plan.eval, &funds, 4096 / threads, players);
                      (*i, rates[0])
                    })
                    .collect()
                })
                .collect();

              let mut network_rates = vec![Vec::new(); networks.len()];

              for rates in results {
                for (i, rate) in rates {
                  network_rates[i].push(rate);
                }
              }

              let mut bench_summary = String::new();

              let mut rates = vec![0.0; networks.len()];

              for (i, xs) in network_rates.iter().enumerate() {
                rates[i] = math::mean(xs);
                bench_summary.push_str(format!(" {:08.0}", rates[i]).as_str());
              }

              // Check if network is the best so far
              let mut network_best = None;
              for (i, (network_p, network_q)) in networks.iter() {
                if rate_max < rates[*i] {
                  network_best = Some(*i);
                  rate_max = rates[*i];
                  debug!("Saving best network {:6} @ {:08.0} ...", i, rates[*i]);
                  let ref network_file = path_training.join("network-p.data");
                  bincode::serialize_into_file(network_file, &network_p).unwrap();
                  let ref network_file = path_training.join("network-q.data");
                  bincode::serialize_into_file(network_file, &network_q).unwrap();
                }
              }

              for i in network_best {
                bench_summary.push_str(format!(" BEST@{}", i).as_str());
              }

              // Logging
              info!(" {:6} |W|", epoch);
              info!(" {:6} |W| {}", epoch, bench_summary);
              info!(" {:6} |W|", epoch);

              // Write in benchmarking file
              let run_summary = format!("{:6} \t {}", epoch, bench_summary);
              benchmarking_log.write(format!("{}\n", run_summary).as_bytes()).unwrap();
              benchmarking_log.flush().unwrap();
            }

            if epoch == epoch_end {
              break;
            }
          }
        }
      }
    }),
    tx,
  )
}
