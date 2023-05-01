#[macro_use]
extern crate log;

extern crate ctrlc;
extern crate nnet;
extern crate rand;
extern crate rayon;
extern crate redis;

extern crate anna_arena;
extern crate anna_eval;
extern crate anna_learning;
extern crate anna_model;
extern crate anna_simulation;
extern crate anna_utils;

use std::{
  sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender},
    Arc,
  },
  thread::JoinHandle,
  time::Duration,
};

use rand::StdRng;
use redis::Client;

use anna_eval::Eval;
use anna_learning::{
  agent_episode::Episode,
  plan::{CardsEncoders, Plan},
  qnet::QNet,
  snet::SNet,
  snet_lex::SNetLex,
};
use anna_model::{
  encoders::{ActionsBinary, ActionsCompact, CardsEncoder},
  ActionClass,
};

use anna_arena::train_worker::{worker_start, Event};

/**
 * train
 *
 * cd bin-arena
 * env RUST_LOG="info" cargo run --release --bin train 0 0 ../resources/ ../resources/training/tl-2
 *
 **/

#[derive(PartialEq, Eq)]
pub enum Mode {
  Online,
  Offline,
  Snapshot { period: Duration },
}

fn main() {
  use rayon::current_num_threads;
  use std::{env::args, path::Path};

  use anna_learning::plan::*;

  let threads = current_num_threads();

  let path_data_str = args().nth(3).unwrap();
  let ref path_data = Path::new(path_data_str.as_str());
  let ref path_synthetic = path_data.join("synthetic");

  // let mode: Mode = Mode::Online;
  let mode: Mode = Mode::Snapshot { period: Duration::from_secs(180) };

  // let plan = plan_kuhn2();
  // let plan = plan_kuhn3();

  // let plan = plan_leduc2();
  // let plan = plan_leduc_french2();
  // let plan = plan_leduc_french3();

  // let plan = plan_cochard2();

  // let plan = plan_texas_limit_n(5, &path_synthetic);
  let plan = plan_texas_limit_zero_n(3, &path_synthetic);

  let (worker, networks_sender) = worker_start(plan.clone(), threads);
  training_run(mode, &plan, threads, &networks_sender).unwrap();
  worker.join().unwrap();
}

fn training_run<'a, AC: ActionClass, E: Eval>(
  mode: Mode,
  plan: &Plan<AC, E>,
  threads: usize,
  worker: &Sender<Event>,
) -> Result<(), nnet::Error>
where
  E: Send,
{
  use rayon::prelude::*;
  use std::{
    cmp::Ordering, collections::HashSet, env::args, io::Write, iter::FromIterator, ops::Neg,
    path::Path, sync::atomic, time::Instant,
  };

  use anna_eval::Eval;
  use anna_learning::{
    agent::{Agent, AgentParams},
    agent_episode::Policy,
    agent_system::AgentSystem,
    qnet::QNet,
    reward::{Reward, RewardF},
  };
  use anna_model::{classifiers::*, profile::*, Money};
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

  let ref mut thread_rng = rand::thread_rng();

  let epoch_init = args().nth(1).unwrap().parse::<usize>().unwrap();
  let epoch_end = args().nth(2).unwrap().parse::<usize>().unwrap();

  let path_data_str = args().nth(3).unwrap();
  let ref path_data = Path::new(path_data_str.as_str());
  let ref path_graphs = path_data.join("graphs");
  let ref path_networks = path_data.join("networks");

  let ref path_training_str = args().nth(4).unwrap();
  let ref path_training = Path::new(path_training_str.as_str());

  let ref learning_log_path = path_training.join("learning.log");

  let learning_log_file = {
    use std::fs::OpenOptions;
    let mut options = OpenOptions::new();
    options.create(true).write(true).append(true);
    options.open(learning_log_path).expect("Unable to open learning log.")
  };

  let mut learning_log = std::io::BufWriter::new(&learning_log_file);

  let fund_init = Money::new(100, 0);

  let ref profile = plan.profile;
  let ref eval = plan.eval;

  let ref reward = plan.reward_f();
  let ref sim = plan.sim();

  let snet = SNet::new(path_graphs, &plan.profile, &plan.eval, &plan.snet_hiddens).unwrap();

  let cards_encoder_create = |rng| {
    match plan.cards_encoders(rng, &snet, &plan.eval, path_networks) {
      CardsEncoders::Binary(encoder) => encoder,
      // CardsEncoders::StrengthNNet(encoder) => encoder,
      _ => unimplemented!(),
    }
  };

  let networks_send =
    |worker: &Sender<Event>, epoch_start, system: &mut AgentSystem, benchmark, save| {
      worker
        .send(Event::Network {
          epoch: epoch_start,
          networks: system
            .agents
            .iter_mut()
            .enumerate()
            .map(|(i, agent)| {
              (i, (agent.p_network.context.get().unwrap(), agent.q_network.context.get().unwrap()))
            })
            .collect(),
          action_benchmark: benchmark,
          action_save: save,
        })
        .unwrap();
    };

  let actions_encoder = plan.actions_encoder();
  let mut cards_encoder = cards_encoder_create(random::rseed(thread_rng));

  let ref qnet =
    QNet::new(path_graphs, sim, actions_encoder.as_ref(), &cards_encoder, &plan.qnet_hiddens)?;

  let redis = redis::Client::open("redis://127.0.0.1/")?;

  let mut system = AgentSystem::new(
    &redis,
    qnet,
    plan.agents,
    plan.agents_hist,
    plan.agents_hist_sampling,
    plan.p_reservoir_min,
    plan.p_reservoir_size,
    plan.q_buffer_size,
  )?;
  let ref agent_params = plan.agent_params;

  let mut agent_avgs = vec![math::MovingAvg::new(plan.agents_avgs); system.agents.len()];

  let mut epoch_start = 0;
  let mut epoch_previous = 0;

  let mut pool_players = Vec::new();
  for _ in 0..threads {
    let mut players = Vec::new();
    for player_id in 0..profile.players {
      let mut rng = random::rseed(thread_rng);
      players.push((
        Episode::new(
          &mut rng,
          actions_encoder.as_ref(),
          cards_encoder_create(random::rseed(thread_rng)),
          qnet.p_predict()?,
          qnet.q_predict()?,
          qnet,
        )?,
        HashSet::new(),
      ));
    }

    pool_players.push(PlayersMulX(players));
  }

  if epoch_init > 0 {
    debug!("Loading snapshot {} ...", epoch_init);
    for i in 0..system.agents.len() {
      let network_p: nnet::Network = {
        let ref file = path_training.join(format!("epoch-{:012}/network-{}-p.data", epoch_init, i));
        bincode::deserialize_from_file(file).expect("Network not found!")
      };

      let network_q: nnet::Network = {
        let ref file = path_training.join(format!("epoch-{:012}/network-{}-p.data", epoch_init, i));
        bincode::deserialize_from_file(file).expect("Network not found!")
      };

      system.agents[i].p_network.context.set(&network_p).unwrap();
      system.agents[i].q_network.context.set(&network_q).unwrap();

      system.agents[i].p_behaviors.snapshot_load();
      system.agents[i].q_transitions.snapshot_load();
    }
  }

  let running = Arc::new(AtomicBool::new(true));

  {
    let running = running.clone();
    ctrlc::set_handler(move || {
      warn!("Interruption requested!");
      running.store(false, atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
  }

  let mut snapshot_start = Instant::now();

  while (running.load(atomic::Ordering::SeqCst) && (epoch_end == 0 || epoch_start < epoch_end)) {
    let mut agent_p_costs: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];
    let mut agent_p_max: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];
    let mut agent_p_min: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];
    let mut agent_q_costs: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];
    let mut agent_q_max: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];
    let mut agent_q_min: Vec<Vec<f32>> = vec![Vec::new(); system.agents.len()];

    // Unroll epochs
    let session_start = Instant::now();
    for epoch in epoch_start..(epoch_start + plan.epochs) {
      trace!("epoch: {:6}", epoch);

      // Run agent system for one epoch
      let (agent_rates, (p_costs, p_maxs, p_mins), (q_costs, q_maxs, q_mins)) = system
        .run_epoch_par(
          &mut pool_players,
          &mut random::rseed(thread_rng),
          sim,
          qnet,
          agent_params,
          actions_encoder.as_ref(),
          &mut cards_encoder,
          reward,
          eval,
          fund_init,
          plan.epoch_games,
          threads,
        )
        .unwrap();

      for (agent_id, agent) in system.agents.iter().enumerate() {
        let ref rates = agent_rates[agent_id];
        if !rates.is_empty() {
          for &rate in rates {
            agent_avgs[agent_id].push(rate);
          }
        }

        agent_p_costs[agent_id].extend(p_costs[agent_id].iter());
        agent_p_max[agent_id].push(p_maxs[agent_id]);
        agent_p_min[agent_id].push(p_mins[agent_id]);
        agent_q_costs[agent_id].extend(q_costs[agent_id].iter());
        agent_q_max[agent_id].push(q_maxs[agent_id]);
        agent_q_min[agent_id].push(q_mins[agent_id]);
      }
    }

    epoch_previous = epoch_start;
    epoch_start += plan.epochs;

    for (agent_id, agent) in system.agents.iter_mut().enumerate() {
      let p_usage = agent.p_behaviors.usage() * 100.0;
      let p_cost: f32 =
        agent_p_costs[agent_id].iter().cloned().sum::<f32>() / agent_p_costs[agent_id].len() as f32;
      let p_max: f32 = agent_p_max[agent_id]
        .iter()
        .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap();
      let p_min: f32 = agent_p_min[agent_id]
        .iter()
        .min_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap();

      let q_usage = agent.q_transitions.usage() * 100.0;
      let q_cost: f32 =
        agent_q_costs[agent_id].iter().cloned().sum::<f32>() / agent_q_costs[agent_id].len() as f32;
      let q_max: f32 = agent_q_max[agent_id]
        .iter()
        .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap();
      let q_min: f32 = agent_q_min[agent_id]
        .iter()
        .min_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap();

      let p_rate = agent_params.p_rate.apply(agent.iterations);
      let q_rate = agent_params.q_rate.apply(agent.iterations);
      let discount_factor = agent_params.discount_factor.apply(agent.iterations);
      let exploration = agent_params.exploration.apply(agent.iterations);

      let learning_summary = format!(
        "{:3} P: η={:.3} {:.1} +{:.1} {:.1} ({:.0}%) Q: η={:.3} {:.1} +{:.1} {:.1} \
                  ({:.0}%) | γ: {:.3} ε: {:.3}",
        agent_id,
        p_rate,
        p_cost,
        p_max,
        p_min,
        p_usage,
        q_rate,
        q_cost,
        q_max,
        q_min,
        q_usage,
        discount_factor,
        exploration
      );

      learning_log.write(format!("{:6} {}\n", epoch_start, learning_summary).as_bytes()).unwrap();
      learning_log.flush().unwrap();
      info!(" {:6} |T| {}", epoch_start, learning_summary);
    }

    let mut perf_summary = String::new();
    for (i, rates) in agent_avgs.iter().enumerate() {
      perf_summary.push_str(format!("{}: {:5.0}\t", i, rates.mean()).as_str());
    }
    debug!("{:6} |T| {}", epoch_start, perf_summary);

    let session_elapsed = session_start.elapsed();
    info!(" {:6} |T| in {:?}", epoch_start, session_start.elapsed());

    if epoch_start >= agent_params.ramping {
      match mode {
        Mode::Online => {
          networks_send(worker, epoch_start, &mut system, true, true);
        }
        Mode::Offline => {
          networks_send(worker, epoch_start, &mut system, false, true);
        }
        Mode::Snapshot { period } => {
          let snapshot_elapsed = snapshot_start.elapsed();
          if snapshot_elapsed >= period {
            networks_send(worker, epoch_start, &mut system, true, true);

            info!(" {:6} |T| Saving snapshot ... ({:?})", epoch_start, snapshot_elapsed);

            for i in 0..system.agents.len() {
              system.agents[i].p_behaviors.snapshot_store();
              system.agents[i].q_transitions.snapshot_store();
            }

            let mut conn = redis.get_connection().unwrap();
            let _: () = redis::cmd("BGSAVE").query(&mut conn).unwrap();

            snapshot_start = Instant::now();
          } else {
            networks_send(worker, epoch_start, &mut system, true, false);
          }
        }
      }
    }
  }

  if running.load(atomic::Ordering::SeqCst) == false {
    worker.send(Event::Abort).unwrap();
  }

  Ok(())
}
