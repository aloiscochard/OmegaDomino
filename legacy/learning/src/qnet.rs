use std::{
  collections::HashSet,
  fmt,
  path::{Path, PathBuf},
};

use nnet::{self, tasks, Graph, Tensor};

use anna_model::{
  cards::Card,
  encoders::{ActionsEncoder, CardsEncoder},
  profile::{Limit, Profile},
  ActionClass,
};

use anna_simulation::{Act, Event, SeatId, Sim};
use anna_utils::bincode;

pub struct QNet<'a> {
  pub action_class: &'a ActionClass,

  pub profile: Profile,
  pub caps: usize,

  pub inputs: usize,
  pub outputs: usize,

  pub hiddens: Vec<usize>,

  pub p_graph_train: Graph,
  pub p_graph_predict: Graph,

  pub q_graph_train: Graph,
  pub q_graph_predict: Graph,
}

impl<'a> QNet<'a> {
  pub fn new(
    graphs_path: &Path,
    sim: &'a Sim,
    actions_encoder: &ActionsEncoder,
    cards_encoder: &CardsEncoder,
    hiddens: &Vec<usize>,
  ) -> Result<QNet<'a>, nnet::Error> {
    let caps = match sim.profile.limit {
      None => panic!("TODO"),
      Some(Limit { caps, .. }) => caps,
    };

    let inputs = actions_encoder.size() + cards_encoder.size();
    let outputs = sim.action_class.size();

    let dims = format!("{}-{:?}-{}", inputs, hiddens, outputs);

    let graph = |kind: &str, mode: &str| -> PathBuf {
      graphs_path.join(format!("qnet-{}-{}-{}.pb", kind, dims, mode))
    };

    let graph_load = |graph: &str| -> Result<Graph, nnet::Error> {
      println!("{:?}", graph);
      nnet::graph_load(graph)
    };

    Ok(QNet {
      action_class: sim.action_class,
      profile: sim.profile.clone(),
      caps: caps,

      inputs: inputs,
      outputs: outputs,
      hiddens: hiddens.clone(),

      p_graph_train: graph_load(&graph("p", "train").to_str().unwrap())?,
      p_graph_predict: graph_load(&graph("p", "predict").to_str().unwrap())?,

      q_graph_train: graph_load(&graph("q", "train").to_str().unwrap())?,
      q_graph_predict: graph_load(&graph("q", "predict").to_str().unwrap())?,
    })
  }

  pub fn encode_action(&self, action: u8) -> Vec<f32> {
    let mut xs: Vec<f32> = vec![0.; self.action_class.size()];
    xs[action as usize] = 1.0;
    xs
  }

  pub fn network_name(&self, profile: &Profile) -> String {
    self.network_name_(&profile.id, profile.players)
  }

  pub fn network_name_(&self, id: &String, players: usize) -> String {
    let dims = format!("{}-{:?}", self.inputs, self.hiddens);
    format!("qnet-{}-{}-{}.data", id, players, dims)
  }

  pub fn network_load(
    &self,
    id: &String,
    players: usize,
    networks_path: &Path,
  ) -> Result<nnet::Network, bincode::Error> {
    use self::bincode::deserialize_from_file;
    deserialize_from_file(&networks_path.join(self.network_name_(id, players)))
  }

  pub fn p_train(&self) -> Result<tasks::Train<f32, f32>, nnet::Error> {
    Ok(tasks::Train::new(nnet::NetworkContext::new(&self.p_graph_train, self.hiddens.len())?))
  }

  pub fn p_predict(&self) -> Result<tasks::Predict<f32, f32>, nnet::Error> {
    Ok(tasks::Predict::new(nnet::NetworkContext::new(&self.p_graph_predict, self.hiddens.len())?))
  }

  pub fn q_train(&self) -> Result<tasks::Train<f32, f32>, nnet::Error> {
    Ok(tasks::Train::new(nnet::NetworkContext::new(&self.q_graph_train, self.hiddens.len())?))
  }

  pub fn q_predict(&self) -> Result<tasks::Predict<f32, f32>, nnet::Error> {
    Ok(tasks::Predict::new(nnet::NetworkContext::new(&self.q_graph_predict, self.hiddens.len())?))
  }

  pub fn run_all_max(
    &self,
    q_target_predict: &mut tasks::Predict<f32, f32>,
    iss: &Vec<Vec<f32>>,
  ) -> Result<Tensor<f32>, nnet::Error> {
    use nnet::tensor_of;
    let mut is = Vec::new();
    for is_ in iss {
      is.extend(is_);
    }

    let is = tensor_of(&[iss.len() as u64, self.inputs as u64], &is)?;
    let os = q_target_predict.run_get::<f32>(&is, "nnet_output_max")?;
    Ok(os)
  }

  pub fn train_many(
    train: &mut tasks::Train<f32, f32>,
    times: usize,
    learning_rate: f32,
    inputs: &Tensor<f32>,
    outputs: &Tensor<f32>,
  ) -> Result<(f32, f32, f32), nnet::Error> {
    use std::cmp::Ordering;

    let mut cost = 0.0;
    let mut gradients_max = Vec::new();
    let mut gradients_min = Vec::new();

    for _ in 0..times {
      let results = train.run_(
        inputs,
        outputs,
        &[(String::from("nnet_learning_rate"), learning_rate)],
        &[
          String::from("nnet_cost"),
          String::from("nnet_gradients_max"),
          String::from("nnet_gradients_min"),
        ],
      )?;

      cost += results[0] / times as f32;
      gradients_max.push(results[1]);
      gradients_min.push(results[2]);
    }

    let max =
      gradients_max.iter().max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap();
    let min =
      gradients_min.iter().min_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap();

    Ok((cost, *max, *min))
  }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct QState {
  pub player: SeatId,
  pub players: HashSet<SeatId>, // Non-folded players
  pub round_id: usize,

  lap: usize,
  lap_players: Vec<usize>, // Players that made an action in the current lap.

  data_cards: Vec<Vec<Card>>,              // cards by rounds
  data_hist: Vec<Vec<(usize, usize, u8)>>, // (round_id, lap, action) by players loc
}

impl QState {
  pub fn empty(qnet: &QNet) -> QState {
    QState {
      player: 0,
      players: HashSet::new(),
      round_id: 0,
      lap: 0,
      lap_players: Vec::new(),
      data_cards: vec![Vec::new(); qnet.profile.rounds.len()],
      data_hist: vec![Vec::new(); qnet.profile.players],
    }
  }

  pub fn new(qnet: &QNet, player: SeatId, player_cards: &Vec<Card>) -> QState {
    let mut state = QState::empty(qnet);

    state.set_cards(&player_cards);
    state.player = player;
    state.players = (0..qnet.profile.players).collect();

    state
  }

  pub fn player_loc(&self, profile: &Profile, seat_id: SeatId) -> usize {
    if seat_id == self.player {
      0
    } else if seat_id > self.player {
      seat_id - self.player
    } else {
      profile.players - (seat_id + 1)
    }
  }

  pub fn update(&mut self, qnet: &QNet, event: &Event) -> () {
    match event {
      &Event::Table { ref cards } => {
        self.round_id += 1;
        self.lap = 0;
        self.lap_players = Vec::new();
        if !cards.is_empty() {
          self.set_cards(cards);
        }
      }
      &Event::Play(Act { seat_id, action }) => {
        let player = self.player_loc(&qnet.profile, seat_id);

        if qnet.action_class.is_fold(action) {
          self.players.remove(&seat_id);
        }

        // Increment lap
        if self.lap_players.contains(&player) {
          self.lap += 1;
          self.lap_players = Vec::new();
        }

        self.lap_players.push(player);

        // Add action in history
        self.data_hist[player].push((self.round_id, self.lap, action));
      }
    }
  }

  fn set_cards(&mut self, cards: &Vec<Card>) -> () {
    self.data_cards[self.round_id].extend(cards);
  }

  pub fn to_vector(
    &self,
    qnet: &QNet,
    actions_encoder: &ActionsEncoder,
    cards_encoder: &mut CardsEncoder,
  ) -> Vec<f32> {
    let players = qnet.profile.players;
    let playing = self.players.len();

    assert!(playing <= players);
    assert!(playing >= 2);

    let mut xs = cards_encoder.encode(&self.data_cards, playing);
    xs.extend(actions_encoder.encode(&self.data_hist));

    xs
  }
}

impl fmt::Debug for QState {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    writeln!(f, "# QState")?;
    writeln!(f, "")?;

    writeln!(f, "## fields")?;
    writeln!(f, "\tplayer: {}", self.player)?;
    writeln!(f, "\tround_id: {}", self.round_id)?;
    writeln!(f, "\tlap: {}", self.lap)?;
    writeln!(f, "\tlap_players: {:?}", self.lap_players)?;
    writeln!(f, "")?;

    writeln!(f, "## cards")?;
    writeln!(f, "{:?}", self.data_cards)?;

    for (i, xs) in self.data_hist.iter().enumerate() {
      writeln!(f, "## player {}", i);
      writeln!(f, "{:?}", xs);
    }
    writeln!(f, "==")?;
    Ok(())
  }
}
