use rand::Rng;
use std::collections::HashMap;

use anna_model::{
  cards::{Card, KUHN_CARDS},
  ActionClass, Money,
};
use players::Players;
use Event;
use SeatId;

// Optimal logic for Kuhn game of 2 players.
// https://upload.wikimedia.org/wikipedia/commons/a/a9/Kuhn_domino_tree.svg

#[derive(Clone)]
pub struct Kuhn2<'a, R: 'a> {
  rng: R,
  action_class: &'a ActionClass,
  alpha: f32,
  alpha_random: bool,
  blind_biggest: Money,
  player_first: SeatId,
  players_acts: Vec<usize>,
  players_card: HashMap<SeatId, Card>,
}

impl<'a, R: Rng> Kuhn2<'a, R> {
  pub fn new(rng: R, ac: &'a ActionClass) -> Kuhn2<'a, R> {
    Kuhn2 {
      rng: rng,
      action_class: ac,
      alpha: 0.0,
      alpha_random: true,
      blind_biggest: Money::zero(),
      player_first: 0,
      players_acts: Vec::new(),
      players_card: HashMap::new(),
    }
  }

  pub fn with_alpha(rng: R, ac: &'a ActionClass, alpha: f32) -> Kuhn2<'a, R> {
    Kuhn2 {
      rng: rng,
      action_class: ac,
      alpha: alpha,
      alpha_random: false,
      blind_biggest: Money::zero(),
      player_first: 0,
      players_acts: Vec::new(),
      players_card: HashMap::new(),
    }
  }
}

impl<'a, R: Rng> Players<()> for Kuhn2<'a, R> {
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    _: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    if self.alpha_random {
      self.alpha = self.rng.gen::<f32>() / 3.0;
    };
    self.blind_biggest = *blinds.iter().max().unwrap();
    self.player_first = player_first;
    self.players_acts = vec![0; blinds.len()];

    for &(seat_id, ref cards) in playing_hands {
      assert!(cards.len() == 1);
      self.players_card.insert(seat_id, cards[0]);
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
    _: &[Event],
  ) -> Result<u8, ()> {
    use anna_utils::math::sample;

    //println!("{}", events.len());

    let card =
      KUHN_CARDS.binary_search(&self.players_card.get(&seat_id).expect("Card not found.")).unwrap();

    let mut probs = vec![0.0; 3];

    if self.player_first == seat_id {
      match card {
        2 => {
          // P1 - K
          if self.players_acts[seat_id] == 0 {
            probs[2] = 3.0 * self.alpha;
            probs[1] = 1.0 - probs[2];
          } else {
            probs[1] = 1.0;
          }
        }
        1 => {
          // P1 - Q
          if self.players_acts[seat_id] == 0 {
            probs[1] = 1.0;
          } else {
            probs[0] = 0.66 - self.alpha;
            probs[1] = 0.33 + self.alpha;
          }
        }
        0 => {
          // P1 - J
          if self.players_acts[seat_id] == 0 {
            probs[2] = self.alpha;
            probs[1] = 1.0 - probs[2];
          } else {
            probs[0] = 1.0;
          }
        }
        _ => panic!("Invalid card {}", card),
      }
    } else {
      for i in self.action_class.normalize(
        self.blind_biggest,
        round_id,
        table_target,
        table_target_raise,
        player_fund,
        player_pots[seat_id],
      ) {
        probs[i] = 1.0;
      }

      match card {
        2 => {
          // P2 - K
          if probs[0] != 0.0 {
            probs[0] = 0.0;
            probs[1] = 1.0;
            probs[2] = 0.0;
          } else {
            probs[0] = 0.0;
            probs[1] = 0.0;
            probs[2] = 1.0;
          }
        }
        1 => {
          // P2 - Q
          if probs[0] != 0.0 {
            probs[0] = 0.66;
            probs[1] = 0.33;
            probs[2] = 0.0;
          } else {
            probs[0] = 0.0;
            probs[1] = 1.0;
            probs[2] = 0.0;
          }
        }
        0 => {
          // P2 - J
          if probs[0] != 0.0 {
            probs[0] = 1.0;
            probs[1] = 0.0;
            probs[2] = 0.0;
          } else {
            probs[0] = 0.0;
            probs[1] = 0.66;
            probs[2] = 0.33;
          }
        }
        _ => panic!("Invalid card {}", card),
      }
    }

    self.players_acts[seat_id] += 1;

    Ok(sample(&mut self.rng, probs) as u8)
  }
}
