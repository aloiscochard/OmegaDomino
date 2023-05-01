use nnet::{self, tasks, Graph, Network, Tensor};

use std::{
  fmt,
  path::{Path, PathBuf},
};

use anna_model::{
  cards::{cards_class, Card},
  encoders::CardsEncoder,
  profile::Profile,
};

use anna_eval::Eval;
use anna_utils::bincode;

pub struct CardsSNet<'a> {
  predicts: Vec<tasks::Predict<'a, f32, f32>>,
  profile: Profile,
  snet: &'a SNet<'a>,
}

impl<'a> CardsSNet<'a> {
  pub fn new(snet: &'a SNet, profile: &Profile, networks_path: &Path) -> CardsSNet<'a> {
    let mut predicts = Vec::new();

    for players in 2..(profile.players + 1) {
      let mut predict = snet.predict().unwrap();
      let network = snet.network_load(&profile.id, players, networks_path).unwrap();
      predict.context.set(&network).unwrap();
      predicts.push(predict);
    }

    CardsSNet { predicts: predicts, profile: profile.clone(), snet }
  }
}

impl<'a> CardsSNet<'a> {
  pub fn run(&mut self, xss: &Vec<Vec<Card>>, players: usize) -> f32 {
    use nnet::tensor_of;
    let is = self.snet.to_vector(&self.profile, xss);
    let input = tensor_of(&[1, self.snet.inputs as u64], &is).unwrap();
    let output = self.predicts[players - 2].run(&input).unwrap();
    output[0]
  }
}

impl<'a> CardsEncoder for CardsSNet<'a> {
  fn encode(&mut self, xss: &Vec<Vec<Card>>, players: usize) -> Vec<f32> {
    let mut rounds_strength = Vec::new();
    rounds_strength.push(self.run(&vec![xss[0].clone()], players));

    for round_id in 1..self.profile.rounds.len() {
      if xss[round_id].len() > 0 {
        rounds_strength.push(self.run(&xss.iter().take(round_id + 1).cloned().collect(), players));
      } else {
        rounds_strength.push(0.0);
      }
    }

    rounds_strength
  }

  fn size(&self) -> usize {
    self.profile.rounds.len()
  }
}

pub struct SNet<'a> {
  pub deck: Vec<Card>,
  pub eval: &'a Eval,
  pub inputs: usize,
  pub hiddens: Vec<usize>,

  pub graph_train: Graph,
  pub graph_predict: Graph,
}

impl<'a> SNet<'a> {
  pub fn new(
    graphs_path: &Path,
    profile: &Profile,
    eval: &'a Eval,
    hiddens: &[usize],
  ) -> Result<SNet<'a>, nnet::Error> {
    let cards = profile.deck.len();
    let inputs = cards * 2;

    let dims = format!("{}-{:?}", inputs, hiddens);

    let graph = |mode: &str| -> PathBuf { graphs_path.join(format!("snet-{}-{}.pb", dims, mode)) };

    let graph_load = |graph: &str| -> Result<Graph, nnet::Error> {
      // TODO Remove!
      println!("{:?}", graph);
      nnet::graph_load(graph)
    };

    Ok(SNet {
      deck: profile.deck.clone(),
      eval: eval,
      inputs: inputs,
      hiddens: hiddens.to_vec(),
      graph_train: graph_load(&graph("train").to_str().unwrap())?,
      graph_predict: graph_load(&graph("predict").to_str().unwrap())?,
    })
  }

  pub fn network_name(&self, profile: &Profile) -> String {
    self.network_name_(&profile.id, profile.players)
  }

  pub fn network_name_(&self, id: &String, players: usize) -> String {
    let dims = format!("{}-{:?}", self.inputs, self.hiddens);
    format!("snet-{}-{}-{}.data", id, players, dims)
  }

  pub fn network_load(
    &self,
    id: &String,
    players: usize,
    networks_path: &Path,
  ) -> Result<Network, bincode::Error> {
    use self::bincode::deserialize_from_file;
    deserialize_from_file(&networks_path.join(self.network_name_(id, players)))
  }

  pub fn network_save(
    &self,
    profile: &Profile,
    networks_path: &Path,
    network: &Network,
  ) -> Result<(), bincode::Error> {
    use self::bincode::serialize_into_file;
    serialize_into_file(&networks_path.join(self.network_name(profile)), network)
  }

  pub fn train(&self) -> Result<tasks::Train<f32, f32>, nnet::Error> {
    Ok(tasks::Train::new(nnet::NetworkContext::new(&self.graph_train, self.hiddens.len())?))
  }

  pub fn predict(&self) -> Result<tasks::Predict<f32, f32>, nnet::Error> {
    Ok(tasks::Predict::new(nnet::NetworkContext::new(&self.graph_predict, self.hiddens.len())?))
  }

  pub fn to_vector(&self, profile: &Profile, xss: &[Vec<Card>]) -> Vec<f32> {
    let mut player_cards = vec![0.0; self.deck.len()];
    let mut table_cards = vec![0.0; self.deck.len()];

    for i in cards_class(&self.deck, &xss[0]) {
      player_cards[i] = 1.0;
    }

    for cards in xss[1..].iter() {
      for i in cards_class(&self.deck, cards) {
        table_cards[i] = 1.0;
      }
    }

    player_cards.extend(table_cards);

    player_cards
  }

  pub fn run(
    train: &mut tasks::Train<f32, f32>,
    learning_rate: f32,
    dropout_rate: f32,
    input: &Tensor<f32>,
    target: &Tensor<f32>,
  ) -> Result<f32, nnet::Error> {
    let results = train.run_(
      input,
      target,
      &[
        (String::from("nnet_learning_rate"), learning_rate),
        (String::from("nnet_dropout_rate"), dropout_rate),
      ],
      &[String::from("nnet_cost")],
    )?;

    Ok(results[0])
  }
}

#[derive(Clone)]
pub struct SNetParams {
  pub learning_rate: f32,
  pub batch_size: usize,
  pub accuracy: usize, // Number of samples generated during Monte-Carlo simulation
}
