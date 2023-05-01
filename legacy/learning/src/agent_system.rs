extern crate cpuprofiler;

use self::cpuprofiler::PROFILER;

use rand::{Rng, StdRng};
use redis::Client;
use std::path::Path;

use nnet::{self, Network};

use agent::{Agent, AgentParams};
use agent_episode::Episode;
use anna_eval::Eval;
use anna_model::{
  encoders::{ActionsEncoder, CardsEncoder},
  Money,
};
use anna_simulation::{perf::Rate, players_mul::PlayersMulX, Sim};
use qnet::QNet;
use reward::RewardF;

use data::CBuffer;

// #[derive(Clone)]
pub struct AgentSystem<'a> {
  pub agents: Vec<Agent<'a>>,
  pub agents_hist: Vec<CBuffer<(Network, Network)>>,
  pub agents_hist_sampling: f32,
}

impl<'a> AgentSystem<'a> {
  pub fn new(
    redis_client: &Client,
    qnet: &'a QNet,
    agents_size: usize,
    agents_hist_size: usize,
    agents_hist_sampling: f32,
    p_reservoir_min: f32,
    p_reservoir_size: usize,
    q_buffer_size: usize,
  ) -> Result<AgentSystem<'a>, nnet::Error> {
    use nnet::{
      network_init,
      tasks::{Predict, Train},
      NetworkContext, Tensor,
    };

    let agent_init = |i: usize| -> Result<Agent<'a>, nnet::Error> {
      use data_redis::{CBuffer, RSampler};

      let mut p_train = qnet.p_train()?;
      let mut q_train = qnet.q_train()?;
      let mut q_target = qnet.q_predict()?;

      let q_network = q_train.context.get()?;
      q_target.context.set(&q_network)?;

      Ok(Agent {
        episodes: 0,
        iterations: 0,

        p_network: p_train,
        p_behaviors: RSampler::new(
          p_reservoir_size,
          p_reservoir_min,
          redis_client,
          format!("b:{}:", i).as_bytes().to_vec(),
        ),
        p_steps: 0,
        q_network: q_train,

        q_network_target: q_target,

        q_network_updates: 0,
        q_transitions: CBuffer::new(
          q_buffer_size,
          redis_client,
          format!("t:{}:", i).as_bytes().to_vec(),
        ),
        q_steps: 0,
      })
    };

    let mut agents = Vec::new();
    for i in 0..agents_size {
      agents.push(agent_init(i)?);
    }

    let agents_hist = agents.iter().map(|agent| CBuffer::new(agents_hist_size)).collect();

    Ok(AgentSystem {
      agents: agents,
      agents_hist: agents_hist,
      agents_hist_sampling: agents_hist_sampling,
    })
  }

  pub fn run_epoch<R, CE>(
    &mut self,
    pool_players: &mut Vec<PlayersMulX<Episode<'a, StdRng, CE>>>,
    rng: &mut R,
    sim: &Sim,
    qnet: &'a QNet,
    params: &AgentParams,
    actions_encoder: &ActionsEncoder,
    cards_encoder: &mut CardsEncoder,
    reward_f: &'a RewardF,
    eval: &Eval,
    funds_init: Money,
    size: usize,
  ) -> Result<
    (Vec<Vec<Rate>>, (Vec<Vec<f32>>, Vec<f32>, Vec<f32>), (Vec<Vec<f32>>, Vec<f32>, Vec<f32>)),
    nnet::Error,
  >
  where
    R: Rng,
    R: Send,
    R: Sync,
    CE: CardsEncoder,
  {
    use rayon::current_num_threads;
    let threads = current_num_threads();
    self.run_epoch_par(
      pool_players,
      rng,
      sim,
      qnet,
      params,
      actions_encoder,
      cards_encoder,
      reward_f,
      eval,
      funds_init,
      size,
      threads,
    )
  }

  pub fn run_epoch_par<R, CE>(
    &mut self,
    pool_players: &mut Vec<PlayersMulX<Episode<'a, StdRng, CE>>>,
    rng: &mut R,
    sim: &Sim,
    qnet: &'a QNet,
    params: &AgentParams,
    actions_encoder: &ActionsEncoder,
    cards_encoder: &mut CardsEncoder,
    reward_f: &'a RewardF,
    eval: &Eval,
    funds_init: Money,
    size: usize,
    workers: usize,
  ) -> Result<
    (Vec<Vec<Rate>>, (Vec<Vec<f32>>, Vec<f32>, Vec<f32>), (Vec<Vec<f32>>, Vec<f32>, Vec<f32>)),
    nnet::Error,
  >
  where
    R: Rng,
    R: Send,
    R: Sync,
    CE: CardsEncoder,
  {
    use nnet::tasks;
    use rand::{
      distributions::{Distribution, Range},
      StdRng,
    };
    use rayon::prelude::*;
    use std::{cmp::Ordering, collections::HashSet, iter::FromIterator};

    use agent_episode::{Episode, EpisodeRecord, Policy};
    use anna_simulation::{
      engine::{table_game_simulate, Score},
      perf::{rate, Rate},
      players_mul::PlayersMulX,
      SeatId,
    };
    use anna_utils::random;

    // PROFILER.lock().unwrap().start("/tmp/training/run_epoch.profile").unwrap();

    let ref player_funds: Vec<(SeatId, Money)> =
      (0..qnet.profile.players).map(|seat_id| (seat_id, funds_init)).collect();

    let pool_size = usize::min(size, workers);
    let pool_rngs: Vec<StdRng> = (0..pool_size).map(|_| random::rseed(rng)).collect();

    let mut games: usize = 0;

    let mut agent_rates: Vec<Vec<Rate>> = vec![Vec::new(); self.agents.len()];
    let mut agent_p_scores: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];
    let mut agent_p_max: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];
    let mut agent_p_min: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];
    let mut agent_q_scores: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];
    let mut agent_q_max: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];
    let mut agent_q_min: Vec<Vec<f32>> = vec![Vec::new(); self.agents.len()];

    let agent_networks: Vec<(Network, Network)> = self
      .agents
      .iter_mut()
      .map(|agent| (agent.p_network.context.get().unwrap(), agent.q_network.context.get().unwrap()))
      .collect();

    while games < size {
      let remainings = size - games;
      // TODO Multiple game per worker!
      // let games_per_worker = usize::min(1, remainings / workers);
      let actives = usize::min(remainings, workers);

      let scores: Vec<(Vec<usize>, usize, Vec<(usize, usize)>, Vec<EpisodeRecord>, Score)> =
        pool_players.par_iter_mut()
        // pool_players.iter_mut()
                    .take(actives)
                    .enumerate()
                    .map(|(worker_id, players)| {
                           let rng = &pool_rngs[worker_id];
                           let ref mut rng = rng.clone();

                           // Sample candidate and opponents from the agents pool and history.
                           let mut agents: Vec<usize> = (0..self.agents.len()).collect();
                           let mut opponents: Vec<(usize, usize)> = Vec::new();

                           let candidate = *rng.choose(&agents).unwrap();

                           rng.shuffle(&mut agents);

                           for agent_id in agents {
                             let hist_len = self.agents_hist[agent_id].values.len();
                             let mut sample = |start| {
                               let end = ((hist_len + 1) as f32 * (1.0 - self.agents_hist_sampling))
                                         as usize;
                               if self.agents_hist_sampling < 1.0 && end > start {
                                 let range = Range::new(start, end);
                                 range.sample(rng)
                               } else {
                                 start
                               }
                             };

                             if agent_id == candidate {
                               if hist_len > 1 {
                                 opponents.push((agent_id, sample(1)));
                               }
                             } else {
                               if hist_len > 1 {
                                 opponents.push((agent_id, sample(0)));
                               } else {
                                 opponents.push((agent_id, 0));
                               }
                             }

                             if opponents.len() == (qnet.profile.players - 1) {
                               break;
                             }
                           }

                           // Shuffle seats assigments
                           let mut seats: Vec<usize> = (0..qnet.profile.players).collect();
                           rng.shuffle(&mut seats);
                           let player_first = *rng.choose(&seats).unwrap();

                           // Setup players
                           for player_id in 0..qnet.profile.players {
                             players.mask_set(player_id,
                                              HashSet::from_iter(vec![seats[player_id]]));

                             let episode = players.at(player_id);

                             let (agent_id, hist_id) = if player_id == 0 {
                               (candidate, 0)
                             } else {
                               opponents[player_id - 1]
                             };

                             let ref agent = self.agents[agent_id];

                             let policy = agent.policy_sample(rng, params);

                             if hist_id > 0 {
                               let hist_len = self.agents_hist[agent_id].values.len();
                               let (ref p_network, ref q_network) =
                                 self.agents_hist[agent_id].values[hist_len - hist_id];

                               let network = match episode.record.policy {
                                 Policy::P => p_network.clone(),
                                 Policy::Q(_) => q_network.clone(),
                               };

                               episode.reset(&network, policy).unwrap();
                             } else {
                              let network = match policy {
                                Policy::P => &agent_networks[agent_id].0,
                                Policy::Q(_) => &agent_networks[agent_id].1,
                              };

                               episode.reset(network, policy).unwrap()
                             }
                           }

                           // Run game.
                           let (_, score) = table_game_simulate(rng,
                                                                sim,
                                                                eval,
                                                                players,
                                                                player_funds,
                                                                player_first).unwrap();

                           (seats,
                            candidate,
                            opponents,
                            players.0.iter().map(|(p, _)| p.record.clone()).collect(),
                            score)
                         })
                    .collect();

      // Increment games count
      games += scores.len();

      // Compute episodes rewards and group per agents
      let mut agent_episodes = vec![Vec::new(); self.agents.len()];
      for (seats, candidate, opponents, episodes, score) in scores {
        let mut funds_finals = score.player_funds;
        funds_finals.sort_by_key(|&(seat_id, _)| seat_id);

        let mut player_rates: Vec<Vec<Rate>> = vec![Vec::new(); qnet.profile.players];
        let funds: Vec<(Money, Money)> = funds_finals
          .into_iter()
          .map(|(seat_id, funds_final)| {
            let rate = rate(sim.blind_biggest, funds_init, funds_final);
            player_rates[seat_id].push(rate);

            (funds_init, funds_final)
          })
          .collect();

        let rewards = reward_f.observe(&funds, &score.winners);

        for player_id in 0..qnet.profile.players {
          let (agent_id, hist_id) =
            if player_id == 0 { (candidate, 0) } else { opponents[player_id - 1] };

          if hist_id == 0 {
            let seat_id = seats[player_id];
            agent_rates[agent_id].extend(player_rates[seat_id].iter());
            agent_episodes[agent_id].push((episodes[player_id].clone(), rewards[seat_id]));
          }
        }
      }

      // Update agents with episodes
      let rngs: Vec<StdRng> = (0..self.agents.len()).map(|_| random::rseed(rng)).collect();
      let scores: Vec<(usize, Option<(f32, f32, f32)>, Option<(f32, f32, f32)>, bool)> = self.agents
            .iter_mut()
            // .par_iter_mut()
            .enumerate()
            .filter_map(|(i, agent)| {
                          let episodes = &agent_episodes[i];
                          if !episodes.is_empty() {
                            let rng = &rngs[i];
                            let ref mut rng = rng.clone();

                            let (p_score, q_score, q_updated) =
                              agent.update(rng, qnet, params, actions_encoder, cards_encoder, i, episodes).unwrap();
                            Some((i, p_score, q_score, q_updated))
                          } else {
                            None
                          }
                        })
            .collect();

      for (agent_id, p_res, q_res, q_updated) in scores {
        for (p_score, max, min) in p_res {
          agent_p_scores[agent_id].push(p_score);
          agent_p_max[agent_id].push(max);
          agent_p_min[agent_id].push(min);
        }
        for (q_score, max, min) in q_res {
          agent_q_scores[agent_id].push(q_score);
          agent_q_max[agent_id].push(max);
          agent_q_min[agent_id].push(min);
        }

        // Update agent history if Q network was updated.
        if q_updated && self.agents_hist_sampling < 1.0 {
          let ref mut agent = self.agents[agent_id];
          let p_network = agent.p_network.context.get()?;
          let q_network = agent.q_network.context.get()?;
          self.agents_hist[agent_id].push(vec![(p_network, q_network)]);
        }
      }
    }

    // PROFILER.lock().unwrap().stop().unwrap();
    //

    let reduce_max = |xs: &Vec<f32>| {
      xs.iter()
        .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap_or(0.0)
    };

    let reduce_min = |xs: &Vec<f32>| {
      xs.iter()
        .min_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap_or(0.0)
    };

    let agent_p_max: Vec<f32> = agent_p_max.iter().map(|xs| reduce_max(xs)).collect();
    let agent_p_min: Vec<f32> = agent_p_min.iter().map(|xs| reduce_min(xs)).collect();
    let agent_q_max: Vec<f32> = agent_q_max.iter().map(|xs| reduce_max(xs)).collect();
    let agent_q_min: Vec<f32> = agent_q_min.iter().map(|xs| reduce_min(xs)).collect();

    Ok((
      agent_rates,
      (agent_p_scores, agent_p_max, agent_p_min),
      (agent_q_scores, agent_q_max, agent_q_min),
    ))
  }
}
