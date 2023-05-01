use std::path::Path;

use agent::AgentParams;
use anna_eval::{strength::*, *};
use anna_model::{
  cards::Card,
  classifiers::*,
  encoders::{ActionsBinary, ActionsCompact, ActionsEncoder, CardsBinary, CardsEncoder},
  hparam::HParam,
  profile::Profile,
  ActionClass, Money,
};
use anna_simulation::Sim;
use anna_utils::random;
use reward::RewardF;
use snet::{CardsSNet, SNet, SNetParams};

#[derive(Clone, Eq, PartialEq)]
pub enum ActionsEncoding {
  Binary,
  Compact,
}

#[derive(Clone)]
pub enum CardsEncoding {
  Binary,
  StrengthMC(usize),
  StrengthNNet,
}

#[derive(Clone)]
pub struct Plan<AC, E> {
  pub action_class: AC,
  pub eval: E,

  pub actions_encoding: ActionsEncoding,
  pub cards_encoding: CardsEncoding,

  pub qnet_hiddens: Vec<usize>,
  pub snet_hiddens: Vec<usize>,

  pub agent_params: AgentParams,
  pub snet_params: SNetParams,

  pub profile: Profile,

  pub agents: usize,
  pub agents_avgs: usize,
  pub agents_hist: usize,
  pub agents_hist_sampling: f32,

  pub epochs: usize,
  pub epoch_games: usize, // Games per Epoch

  pub p_reservoir_size: usize,
  pub p_reservoir_min: f32,

  pub q_buffer_size: usize,
}

impl<AC, E> Plan<AC, E>
where
  AC: ActionClass,
{
  pub fn actions_encoder(&self) -> Box<ActionsEncoder> {
    if self.actions_encoding == ActionsEncoding::Binary {
      Box::new(ActionsBinary::new(&self.profile, &self.action_class))
    } else {
      Box::new(ActionsCompact::new(ActionsBinary::new(&self.profile, &self.action_class)))
    }
  }
}

pub enum CardsEncoders<'a, E: Eval + Send + Sync> {
  Binary(CardsBinary),
  StrengthMC(CardsStrength<'a, E>),
  StrengthNNet(CardsSNet<'a>),
}

impl<AC, E> Plan<AC, E> {
  pub fn blind_biggest(&self) -> Money {
    *self.profile.blinds.iter().max().expect("No blinds defined.")
  }

  pub fn delta_biggest(&self) -> Money {
    self.blind_biggest() * self.profile.delta_biggest_r()
  }

  pub fn cards_encoders<'a>(
    &self,
    rng: rand::StdRng,
    snet: &'a SNet,
    eval: &'a E,
    networks_path: &Path,
  ) -> CardsEncoders<'a, E>
  where
    E: Eval + Send + Sync,
  {
    match self.cards_encoding {
      CardsEncoding::Binary => {
        CardsEncoders::Binary(CardsBinary::new(&self.profile.deck, &self.profile.rounds))
      }
      CardsEncoding::StrengthMC(accuracy) => CardsEncoders::StrengthMC(CardsStrength::new(
        rng,
        eval.clone(),
        self.profile.clone(),
        accuracy,
      )),
      CardsEncoding::StrengthNNet => {
        CardsEncoders::StrengthNNet(CardsSNet::new(snet, &self.profile, networks_path))
      }
    }
  }

  pub fn reward_f(&self) -> RewardF {
    RewardF::new(self.delta_biggest())
  }

  pub fn sim(&self, strict: bool) -> Sim
  where
    AC: ActionClass,
  {
    Sim {
      action_class: &self.action_class,
      blind_biggest: self.blind_biggest(),
      profile: self.profile.clone(),
      strict: strict
    }
  }
}

pub fn plan_kuhn2() -> Plan<ActionKuhn, EvalNaive> {
  use anna_model::profile::profile_kuhn;

  let epochs = 200;

  let action_class = ActionKuhn {};
  let profile = profile_kuhn(2);
  let eval = Eval::naive();

  Plan {
    action_class: action_class,
    actions_encoding: ActionsEncoding::Compact,
    cards_encoding: CardsEncoding::StrengthNNet,
    qnet_hiddens: vec![16],
    snet_hiddens: vec![8],

    agent_params: AgentParams {
      anticipation: 0.1,
      discount_factor: HParam::linear(0.8, 0.99, 4000),
      exploration: HParam::decay(0.06),
      steps: 8,
      q_updates_refit: 32,
      batch_size: 32,
      batch_updates: 2,
      p_rate: HParam::new(0.1),
      q_rate: HParam::new(0.1),
      ramping: 0,
    },
    snet_params: SNetParams { learning_rate: 0.1, batch_size: 64, accuracy: 64 },

    agents: 2,
    agents_avgs: 512,

    agents_hist: 1,
    agents_hist_sampling: 1.0,

    epochs: epochs,
    epoch_games: 8,

    p_reservoir_min: 0.0,
    p_reservoir_size: 1_000_000,
    q_buffer_size: 1_000_000,

    profile: profile,
    eval: eval,
  }
}

pub fn plan_kuhn3() -> Plan<ActionKuhn, EvalNaive> {
  use anna_model::profile::profile_kuhn;

  let profile = profile_kuhn(3);
  let plan = plan_kuhn2();

  Plan {
    profile: profile,
    // snet_hiddens: vec![16],
    // snet_params: SNetParams {
    //    learning_rate: 0.01,
    //    batch_size: 256,
    //    accuracy: 256,
    // },
    agents: 3,
    agents_avgs: 1024,
    epoch_games: 16,
    ..plan
  }
}

pub fn plan_leduc2() -> Plan<ActionLimit, EvalNaive> {
  use anna_model::profile::profile_leduc;

  let profile = profile_leduc(2);
  let eval = Eval::naive();

  let epochs = 200;

  Plan {
    action_class: ActionLimit { raises: profile.limit.clone().unwrap().raises },
    actions_encoding: ActionsEncoding::Compact,
    cards_encoding: CardsEncoding::StrengthNNet,
    qnet_hiddens: vec![32],
    snet_hiddens: vec![8],
    agent_params: AgentParams {
      anticipation: 0.1,
      discount_factor: HParam::linear(0.6, 0.99, 300_000),
      exploration: HParam::linear(0.1, 0.06, 300_000),
      steps: 128,
      q_updates_refit: 128,
      batch_size: 128,
      batch_updates: 2,
      p_rate: HParam::new(0.05),
      q_rate: HParam::new(0.05),
      ramping: 0,
    },

    snet_params: SNetParams { learning_rate: 0.1, batch_size: 128, accuracy: 128 },

    agents: 2,
    agents_avgs: 1024,

    agents_hist: 1,
    agents_hist_sampling: 1.0,

    epochs: epochs,
    epoch_games: 16,

    p_reservoir_min: 0.0,
    p_reservoir_size: 1_000_000,
    q_buffer_size: 1_000_000,

    profile: profile,
    eval: eval,
  }
}

pub fn plan_leduc_french2() -> Plan<ActionLimit, EvalNaive> {
  use anna_model::profile::profile_leduc_french;

  let profile = profile_leduc_french(2);
  let plan = plan_leduc2();

  Plan {
    profile: profile,
    qnet_hiddens: vec![32],
    snet_hiddens: vec![16],
    agent_params: AgentParams {
      discount_factor: HParam::linear(0.6, 0.99, 600_000),
      exploration: HParam::linear(0.1, 0.06, 600_000),
      ..plan.agent_params
    },
    snet_params: SNetParams { learning_rate: 0.01, batch_size: 256, accuracy: 256 },
    p_reservoir_size: 2_000_000,
    q_buffer_size: 200_000,
    ..plan
  }
}

pub fn plan_leduc_french3() -> Plan<ActionLimit, EvalNaive> {
  use anna_model::profile::profile_leduc_french;

  let profile = profile_leduc_french(3);
  let plan = plan_leduc_french2();

  Plan { profile: profile, agents: 3, ..plan }
}

pub fn plan_leduc_french6() -> Plan<ActionLimit, EvalNaive> {
  use anna_model::profile::profile_leduc_french;

  let profile = profile_leduc_french(6);
  let plan = plan_leduc_french2();

  Plan { profile: profile, agents: 6, ..plan }
}

pub fn plan_cochard2() -> Plan<ActionLimit, EvalNaive> {
  use anna_model::profile::profile_cochard;

  let profile = profile_cochard(2);
  let plan = plan_leduc_french2();

  Plan {
    action_class: ActionLimit { raises: profile.limit.clone().unwrap().raises },
    qnet_hiddens: vec![32],
    snet_hiddens: vec![512],
    snet_params: SNetParams { learning_rate: 0.001, batch_size: 512, accuracy: 1024 },
    cards_encoding: CardsEncoding::StrengthNNet,
    profile: profile,
    ..plan
  }
}

pub fn plan_texas_limit_n(n: usize, path_synthetic: &Path) -> Plan<ActionLimit, EvalTexas> {
  // pub fn plan_texas_limit_n(n: usize, path_synthetic: &Path) ->
  // Plan<ActionLimit, EvalTexasCache> {
  use anna_model::profile::profile_texas_limit;

  let epochs = 200 * n;

  let blind_big = Money::new(1, 0);
  let blind_small = blind_big / 2;

  let profile = profile_texas_limit(n, blind_small, blind_big);
  // let eval = Eval::texas_cache(path_synthetic);
  let eval = Eval::texas();

  Plan {
    action_class: ActionLimit { raises: profile.limit.clone().unwrap().raises },
    actions_encoding: ActionsEncoding::Compact,
    cards_encoding: CardsEncoding::StrengthNNet,
    qnet_hiddens: vec![1024],
    snet_hiddens: vec![512],

    agent_params: AgentParams {
      anticipation: 0.1,
      discount_factor: HParam::linear(0.6, 0.99, 900_000),
      exploration: HParam::linear(0.1, 0.06, 900_000),
      steps: 128,
      q_updates_refit: 128,
      batch_size: 128,
      batch_updates: 2,
      p_rate: HParam::new(0.05),
      q_rate: HParam::new(0.05),
      ramping: 0,
    },

    snet_params: SNetParams { learning_rate: 0.001, batch_size: 1024, accuracy: 2048 },

    agents: n,
    agents_avgs: 1024,

    agents_hist: 1,
    agents_hist_sampling: 1.0,

    epochs: epochs,
    epoch_games: 16,

    p_reservoir_min: 0.0,
    p_reservoir_size: 6_000_000, // / n,
    q_buffer_size: 300_000,

    profile: profile,
    eval: eval,
  }
}

pub fn plan_texas_limit_zero_n(n: usize, path_synthetic: &Path) -> Plan<ActionLimit, EvalTexas> {
  let plan = plan_texas_limit_n(n, path_synthetic);
  let profile = Profile { id: "texas_limit_zero".to_string(), ..plan.profile };

  Plan {
    profile: profile,
    actions_encoding: ActionsEncoding::Binary,
    cards_encoding: CardsEncoding::Binary,
    qnet_hiddens: vec![1024, 512],

    agent_params: AgentParams {
      exploration: HParam::linear(0.08, 0.01, 10_000_000),

      steps: 256,
      q_updates_refit: 1000,
      batch_size: 256,

      ..plan.agent_params
    },

    p_reservoir_size: 30_000_000,
    q_buffer_size: 600_000,

    ..plan
  }
}
