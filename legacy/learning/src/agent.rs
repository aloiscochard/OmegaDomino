use rand::{Rng, StdRng};

use nnet::{tasks, Network, Tensor};

use anna_model::{
  cards::Card,
  encoders::{ActionsEncoder, CardsEncoder},
  hparam::HParam,
  Money,
};
use anna_simulation::{players::Players, Event, SeatId};
use qnet::{QNet, QState};
use reward::{Reward, RewardF};

use agent_episode::{EpisodeRecord, Policy};
use data::sample_indices;
use data_redis::{CBuffer, RSampler, RedisStore};

pub struct Agent<'a> {
  pub episodes: usize,
  pub iterations: usize,

  pub p_network: tasks::Train<'a, f32, f32>,
  pub p_behaviors: RSampler<(QState, u8)>, // (s, a)
  pub p_steps: usize,

  pub q_network: tasks::Train<'a, f32, f32>,
  pub q_network_target: tasks::Predict<'a, f32, f32>,

  pub q_network_updates: usize,
  pub q_transitions: CBuffer<(QState, u8, f32, Option<QState>)>, // (s, a, r, s')
  pub q_steps: usize,
}
unsafe impl<'a> Send for Agent<'a> {}
unsafe impl<'a> Sync for Agent<'a> {}

impl<'a> Agent<'a> {
  pub fn policy_sample<R: Rng>(&self, rng: &mut R, params: &AgentParams) -> Policy {
    if rng.gen::<f32>() < params.anticipation {
      Policy::Q(params.exploration.apply(self.iterations))
    } else {
      Policy::P
    }
  }

  pub fn update<R: Rng>(
    &mut self,
    rng: &mut R,
    qnet: &QNet,
    params: &AgentParams,
    actions_encoder: &ActionsEncoder,
    cards_encoder: &mut CardsEncoder,
    agent_id: usize,
    records: &Vec<(EpisodeRecord, Reward)>,
  ) -> Result<(Option<(f32, f32, f32)>, Option<(f32, f32, f32)>, bool), nnet::Error> {
    use behavior::BehaviorSet;
    use nnet::tensor_of;

    // Compute long term reward per transition using the target network.
    fn transitions_rewards_l(
      qnet: &QNet,
      actions_encoder: &ActionsEncoder,
      cards_encoder: &mut CardsEncoder,
      q_predict: &mut tasks::Predict<f32, f32>,
      transitions: &mut CBuffer<(QState, u8, f32, Option<QState>)>,
      indices: &[usize],
    ) -> Result<Vec<f32>, nnet::Error> {
      let mut is: Vec<Option<usize>> = Vec::new();
      let mut ss: Vec<Vec<f32>> = Vec::new();

      for &i in indices {
        match transitions.get(i).3 {
          Some(ref s) => {
            is.push(Some(ss.len()));
            let xs = s.to_vector(qnet, actions_encoder, cards_encoder);
            ss.push(xs);
          }
          None => {
            is.push(None);
          }
        }
      }

      if ss.len() > 0 {
        let vs = qnet.run_all_max(q_predict, &ss)?;

        let mut rs: Vec<f32> = Vec::new();
        for i in 0..indices.len() {
          rs.push(is[i].map(|j| vs[j]).unwrap_or(0.0));
        }

        Ok(rs)
      } else {
        Ok(vec![0.0; is.len()])
      }
    }

    self.episodes += records.len();
    self.iterations += 1;

    // Update memories
    for &(ref record, reward) in records {
      let behaviors_len = record.behaviors.len();

      if behaviors_len > 0 {
        // Extract and store transitions.
        let transitions: Vec<(QState, u8, f32, Option<QState>)> = record
          .behaviors
          .iter()
          .enumerate()
          .map(|(i, &(ref state, action, _))| {
            let reward_i = if i == (behaviors_len - 1) { reward } else { 0.0 };

            let state_next = {
              if i == behaviors_len - 1 {
                None
              } else {
                let (ref s, _, _) = record.behaviors[i + 1];
                Some(s.clone())
              }
            };

            (state.clone(), action, reward_i, state_next)
          })
          .collect();

        let transitions_len = transitions.len();
        self.q_transitions.push(transitions);
        self.q_steps += transitions_len;

        match record.policy {
          Policy::P => {
            // NOOP
          }
          Policy::Q(_) => {
            // Feed the reservoir
            let xs =
              record.behaviors.iter().map(|&(ref is, action, _)| (is.clone(), action)).collect();

            self.p_steps += self.p_behaviors.push(rng, xs);
          }
        }
      }
    }

    // Train P
    let result_p = if self.p_steps > params.steps && self.iterations >= params.ramping {
      trace!("{} - P - {}", agent_id, self.p_steps);

      self.p_steps = 0;

      let training = {
        let indices = sample_indices(rng, self.p_behaviors.len(), params.batch_size);
        let size = indices.len();

        let mut is = Vec::new();
        let mut os = Vec::new();

        for i in indices {
          let value = self.p_behaviors.get(i);
          is.extend(value.0.to_vector(qnet, actions_encoder, cards_encoder));
          os.extend(qnet.encode_action(value.1));
        }

        let is = tensor_of(&[size as u64, qnet.inputs as u64], &is)?;
        let os = tensor_of(&[size as u64, qnet.outputs as u64], &os)?;

        BehaviorSet { inputs: is, outputs: os, size: size }
      };

      self.p_network.context.optimizer_reset()?;

      Some(QNet::train_many(
        &mut self.p_network,
        params.batch_updates,
        params.p_rate.apply(self.iterations),
        &training.inputs,
        &training.outputs,
      )?)
    } else {
      None
    };

    // Train Q
    let result_q = if self.q_steps > params.steps && self.iterations >= params.ramping {
      trace!("{} - Q - {}", agent_id, self.q_steps);

      self.q_steps = 0;
      self.q_network_updates += 1;

      let indices = sample_indices(rng, self.q_transitions.len(), params.batch_size);
      let size = indices.len();

      let rls: Vec<f32> = transitions_rewards_l(
        qnet,
        actions_encoder,
        cards_encoder,
        &mut self.q_network_target,
        &mut self.q_transitions,
        &indices,
      )?;

      let mut is = Vec::new();
      for &i in indices.iter() {
        let xs = self.q_transitions.get(i).0.to_vector(qnet, actions_encoder, cards_encoder);
        is.extend(xs);
      }
      let is = tensor_of(&[size as u64, qnet.inputs as u64], &is)?;

      let mut os: Tensor<f32> = self.q_network.predict(&is)?;

      let training = {
        for (i, &t_i) in indices.iter().enumerate() {
          let (_, action, ri, _) = self.q_transitions.get(t_i);

          let rl = rls[i];
          let r = ri + (params.discount_factor.apply(self.iterations) * rl);

          let actions_size = qnet.action_class.size();
          for a in 0..actions_size {
            if a == action as usize {
              os[a + (actions_size * i)] = r;
            }
          }
        }

        BehaviorSet { inputs: is, outputs: os, size: size }
      };

      self.q_network.context.optimizer_reset()?;

      Some(QNet::train_many(
        &mut self.q_network,
        params.batch_updates,
        params.q_rate.apply(self.iterations),
        &training.inputs,
        &training.outputs,
      )?)
    } else {
      None
    };

    // Periodically update Q target network.
    let update_q = self.q_network_updates == (params.q_updates_refit - 1);
    if update_q {
      trace!("{} - Q Target", agent_id);

      self.q_network_updates = 0;
      self.q_network_target.context.set(&self.q_network.context.get()?)?;
    }

    Ok((result_p, result_q, update_q))
  }
}

#[derive(Clone)]
pub struct AgentParams {
  pub anticipation: f32, // eta
  pub batch_size: usize,
  pub batch_updates: usize, // Number of SGD updates per mini-batch

  pub discount_factor: HParam, // 𝛾
  pub exploration: HParam,     // ε

  pub steps: usize,           // per epoch
  pub q_updates_refit: usize, // updates per reffit

  pub p_rate: HParam,
  pub q_rate: HParam,

  pub ramping: usize, // Number of epochs used to bootstrap the agent memory (no training)
}
