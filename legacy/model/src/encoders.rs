use cards::{cards_class, Card};
use profile::{Limit, Profile};
use ActionClass;

pub trait ActionsEncoder: Send + Sync {
  // hist: (round_id, lap, action) by players
  fn encode(&self, hist: &Vec<Vec<(usize, usize, u8)>>) -> Vec<f32>;
  fn size(&self) -> usize;
}

#[derive(Clone)]
pub struct ActionsBinary {
  pub players: usize,
  pub actions: usize,
  pub rounds: usize,
  pub laps: usize,

  pub size: usize,
}

impl ActionsBinary {
  pub fn new(profile: &Profile, action_class: &ActionClass) -> ActionsBinary {
    let laps = match profile.limit {
      None => panic!("TODO"),
      Some(Limit { caps, .. }) => caps,
    };

    ActionsBinary {
      players: profile.players,
      actions: action_class.size(),
      rounds: profile.rounds.len(),
      laps: laps,
      size: profile.players * action_class.size() * profile.rounds.len() * laps,
    }
  }
}

impl ActionsEncoder for ActionsBinary {
  fn encode(&self, hist: &Vec<Vec<(usize, usize, u8)>>) -> Vec<f32> {
    let mut xs = vec![0.0; self.size];

    for player_id in 0..self.players {
      let player_idx = player_id * (self.actions * self.rounds * self.laps);
      for &(round_id, lap, action) in hist[player_id].iter() {
        let idx =
          player_idx + (action as usize * (self.rounds * self.laps)) + (round_id * self.laps) + lap;
        xs[idx] = 1.0;
      }
    }

    xs
  }

  fn size(&self) -> usize {
    self.size
  }
}

pub struct ActionsCompact {
  pub source: ActionsBinary,
  pub size: usize,
}

impl ActionsCompact {
  pub fn new(source: ActionsBinary) -> ActionsCompact {
    ActionsCompact { size: 2 * source.actions * source.rounds, source: source }
  }
}

impl ActionsEncoder for ActionsCompact {
  fn encode(&self, hist: &Vec<Vec<(usize, usize, u8)>>) -> Vec<f32> {
    // actions per rounds per players
    let mut hist_dense: Vec<Vec<Vec<u8>>> = vec![vec![vec![]; self.source.rounds]; 2];

    for (player, xs) in hist.iter().enumerate() {
      let i = if player == 0 { 0 } else { 1 };
      for (round_id, _, action) in xs.iter() {
        hist_dense[i][*round_id].push(*action);
      }
    }

    fn actions_dist(n: usize, actions: &[u8]) -> Vec<f32> {
      let mut probs = vec![0.0; n];

      let x = 1.0 / actions.len() as f32;
      for action in actions {
        probs[*action as usize] += x;
      }

      probs
    }

    let mut xs: Vec<f32> = Vec::new();

    for ys in hist_dense {
      for actions in ys {
        xs.extend(actions_dist(self.source.actions, &actions));
      }
    }

    xs
  }

  fn size(&self) -> usize {
    self.size
  }
}

pub trait CardsEncoder: Send + Sync {
  // cards per round, where xss[0] represents private cards.
  fn encode(&mut self, xss: &Vec<Vec<Card>>, player_actives: usize) -> Vec<f32>;
  fn size(&self) -> usize;
}

#[derive(Clone)]
pub struct CardsBinary {
  pub deck: Vec<Card>,
  pub rounds: Vec<usize>,
  pub size: usize,
}

impl CardsBinary {
  pub fn new(deck: &[Card], rounds: &[usize]) -> CardsBinary {
    CardsBinary { size: rounds.len() * deck.len(), rounds: rounds.to_vec(), deck: deck.to_vec() }
  }
}

impl CardsEncoder for CardsBinary {
  fn encode(&mut self, xss: &Vec<Vec<Card>>, players_actives: usize) -> Vec<f32> {
    let mut yss = vec![vec![0.0; self.deck.len()]; self.rounds.len()];

    for (round_id, cards) in xss.iter().enumerate() {
      for i in cards_class(&self.deck, cards) {
        yss[round_id][i] = 1.0;
      }
    }

    let mut values = Vec::new();
    for ys in yss {
      values.extend(ys);
    }

    values
  }

  fn size(&self) -> usize {
    self.size
  }
}
