use rand::{thread_rng, Rng, StdRng};
use std::{
  collections::{HashMap, HashSet},
  path::Path,
};

use nnet::{tasks, Network, Tensor};

use anna_model::{
  cards::Card,
  encoders::{ActionsEncoder, CardsEncoder},
  hparam::HParam,
  Money,
};
use anna_simulation::{players::Players, players_sel::PlayersSel, Event, SeatId};
use anna_utils::random;

use qnet::{QNet, QState};
use reward::{Reward, RewardF};

use data::{sample_indices, CBuffer, RSampler};

// This is pretty limited, as it works only for plan that have constant input size
pub fn players_create<'a, CE: CardsEncoder + Send + Sync, F>(
  networks_path: &Path,
  qnet: &'a QNet,
  actions_encoder: &'a ActionsEncoder,
  cards_encoder_create: F,
) -> PlayersSel<Episode<'a, StdRng, CE>>
where
  F: Fn(StdRng) -> CE,
{
  let ref mut thread_rng = rand::thread_rng();
  let mut episodes = Vec::new();

  for p in 2..(qnet.profile.players + 1) {
    let network = qnet.network_load(&qnet.profile.id, p, networks_path).unwrap();

    let mut rng = random::rseed(thread_rng);
    let mut episode = Episode::new(
      &mut rng,
      actions_encoder,
      cards_encoder_create(random::rseed(thread_rng)),
      qnet.p_predict().unwrap(),
      qnet.q_predict().unwrap(),
      &qnet,
    )
    .unwrap();

    episode.reset(&network, Policy::P).unwrap();
    episodes.push(episode);
  }

  PlayersSel::new(episodes)
}

pub fn players_clean<'a, CE: CardsEncoder>(
  players_sel: &mut PlayersSel<Episode<'a, StdRng, CE>>,
) -> () {
  for p in players_sel.players.iter_mut() {
    p.record.behaviors = Vec::new();
  }
}

pub struct Episode<'a, R, CE: CardsEncoder + Send + Sync> {
  pub rng: R,
  pub actions_encoder: &'a ActionsEncoder,
  pub cards_encoder: CE,
  pub p_predict: tasks::Predict<'a, f32, f32>,
  pub q_predict: tasks::Predict<'a, f32, f32>,
  pub qnet: &'a QNet<'a>,
  pub record: EpisodeRecord,
  blind_biggest: Money,
  qstate: QState,
}

impl<'a, R: Rng, CE: CardsEncoder> Episode<'a, R, CE> {
  pub fn new(
    rng: &mut R,
    actions_encoder: &'a ActionsEncoder,
    cards_encoder: CE,
    p_predict: tasks::Predict<'a, f32, f32>,
    q_predict: tasks::Predict<'a, f32, f32>,
    qnet: &'a QNet,
  ) -> Result<Episode<'a, StdRng, CE>, nnet::Error> {
    use anna_utils::random;

    Ok(Episode {
      rng: random::rseed(rng),
      actions_encoder: actions_encoder,
      cards_encoder: cards_encoder,
      p_predict: p_predict,
      q_predict: q_predict,
      qnet: qnet,
      blind_biggest: Money::zero(),
      qstate: QState::empty(qnet),
      record: EpisodeRecord { policy: Policy::P, behaviors: Vec::new() },
    })
  }

  pub fn reset(&mut self, network: &Network, policy: Policy) -> Result<(), nnet::Error> {
    self.record = EpisodeRecord { behaviors: Vec::new(), policy: policy };

    let context = match self.record.policy {
      Policy::P => &mut self.p_predict.context,
      Policy::Q(_) => &mut self.q_predict.context,
    };

    context.set(network)
  }
}

impl<'a, R: Rng, CE: CardsEncoder> Players<()> for Episode<'a, R, CE> {
  fn init(
    &mut self,
    blinds: &[Money],
    _: SeatId,
    _: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    assert!(playing_hands.len() == 1);
    self.blind_biggest = *blinds.iter().max().unwrap();
    for &(seat_id, ref cards) in playing_hands {
      self.qstate = QState::new(self.qnet, seat_id, cards);
    }
  }

  fn play(
    &mut self,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_pots: &[Money],
    seat_id: usize,
    player_fund: Money,
    events: &[Event],
  ) -> Result<u8, ()> {
    use anna_utils::math::sample;
    use nnet::tensor1;
    use std::cmp::Ordering;

    assert!(self.qstate.player == seat_id);

    // Update state
    for event in events {
      self.qstate.update(self.qnet, event);
    }

    assert!(self.qstate.round_id == round_id);

    let is_v = self.qstate.to_vector(self.qnet, self.actions_encoder, &mut self.cards_encoder);
    let is = tensor1(&is_v).unwrap();

    let predict = match self.record.policy {
      Policy::P => &mut self.p_predict,
      Policy::Q(_) => &mut self.q_predict,
    };

    let probs = predict.run(&is).unwrap();

    // Normalize probabilities
    let probs_mask = self.qnet.action_class.normalize(
      self.blind_biggest,
      round_id,
      table_target,
      table_target_raise,
      player_fund,
      player_pots[seat_id],
    );

    let (i, _) = probs
      .iter()
      .enumerate()
      .filter(|&(i, _)| probs_mask.contains(&i))
      .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
      .expect("At least one probs should be > 0.");

    let explore = match self.record.policy {
      Policy::P => false,
      Policy::Q(exploration) => self.rng.gen::<f32>() < exploration,
    };

    // Compute action
    let action = if explore {
      let mut xs: Vec<f32> = vec![0.; probs.len()];
      for j in 0..probs.len() {
        if probs_mask.contains(&j) && j != i {
          xs[j] = 1.0;
        }
      }
      sample(&mut self.rng, xs) as u8
    } else {
      i as u8
    };

    self.record.behaviors.push((self.qstate.clone(), action, explore));

    Ok(action)
  }
}

#[derive(Clone)]
pub struct EpisodeRecord {
  pub policy: Policy,
  pub behaviors: Vec<(QState, u8, bool)>,
}

#[derive(Clone)]
pub enum Policy {
  P,      // SL - Average
  Q(f32), // RL - Best        (exploration)
}
