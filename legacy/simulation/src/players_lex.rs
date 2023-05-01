use rand::Rng;
use std::collections::HashMap;

use anna_eval::Eval;
use anna_model::{cards::Card, profile::Profile, ActionClass, Money};
use players::Players;
use Event;
use SeatId;

#[derive(Clone)]
pub struct Lex<'a, R: 'a> {
  rng: R,
  accuracy: usize,
  action_class: &'a ActionClass,
  eval: &'a Eval,
  profile: &'a Profile,
  blind_biggest: Money,
  // player_first: SeatId,
  players_cards: HashMap<SeatId, Vec<Card>>,
  table_cards: Vec<Card>,
}

impl<'a, R: Rng> Lex<'a, R> {
  pub fn new(
    rng: R,
    accuracy: usize,
    ac: &'a ActionClass,
    eval: &'a Eval,
    profile: &'a Profile,
  ) -> Lex<'a, R> {
    Lex {
      rng: rng,
      accuracy: accuracy,
      action_class: ac,
      eval: eval,
      profile: profile,
      blind_biggest: Money::zero(),
      // player_first: 0,
      players_cards: HashMap::new(),
      table_cards: Vec::new(),
    }
  }
}

impl<'a, R: Rng> Players<()> for Lex<'a, R> {
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    _: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.blind_biggest = *blinds.iter().max().unwrap();
    // self.player_first = player_first;

    for &(seat_id, ref cards) in playing_hands {
      self.players_cards.insert(seat_id, cards.clone());
    }

    self.table_cards = Vec::new();
  }

  fn play(
    &mut self,
    _: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_pots: &[Money],
    seat_id: usize,
    _: Money,
    events: &[Event],
  ) -> Result<u8, ()> {
    use anna_eval::strength::{return_rate, strength};
    use anna_utils::math;

    for event in events {
      match event {
        &Event::Table { ref cards } => {
          self.table_cards.extend(cards.iter());
        }
        _ => {}
      }
    }

    let hand = self.players_cards.get(&seat_id).unwrap();
    let strength =
      strength(&mut self.rng, self.eval, self.profile, self.accuracy, hand, &self.table_cards);
    let target = Money::from_i32(table_target - player_pots[seat_id]).unwrap();
    let raise = table_target_raise.unwrap_or(Money::zero());
    let return_rate = return_rate(strength, player_pots, target + raise);

    let probs = if return_rate < 0.8 {
      vec![0.95, 0.00, 0.05]
    } else if return_rate < 1.0 {
      vec![0.80, 0.05, 0.15]
    } else if return_rate < 1.3 {
      vec![0.00, 0.60, 0.40]
    } else {
      vec![0.00, 0.30, 0.70]
    };

    let action_i = math::sample(&mut self.rng, probs);

    // If fold and amount to call is zero, then call.
    if action_i == 0 && target == Money::zero() {
      Ok(1)
    } else if action_i == 2 && table_target_raise.is_none() {
      Ok(1)
    } else {
      Ok(action_i as u8)
    }
  }
}
