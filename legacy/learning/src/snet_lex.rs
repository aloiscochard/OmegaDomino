use rand::Rng;
use std::collections::{HashMap, HashSet};

use anna_eval::Eval;
use anna_model::{cards::Card, profile::Profile, ActionClass, Money};
use anna_simulation::{players::Players, Act, Event, SeatId};

use snet::CardsSNet;

pub struct SNetLex<'a, R: 'a> {
  rng: R,
  cards_snet: CardsSNet<'a>,
  action_class: &'a ActionClass,
  profile: &'a Profile,
  blind_biggest: Money,
  players: HashSet<SeatId>, // Non-folded players
  players_cards: HashMap<SeatId, Vec<Card>>,
  table_cards: Vec<Card>,
}

impl<'a, R: Rng> SNetLex<'a, R> {
  pub fn new(
    rng: R,
    cards_snet: CardsSNet<'a>,
    ac: &'a ActionClass,
    profile: &'a Profile,
  ) -> SNetLex<'a, R> {
    SNetLex {
      rng: rng,
      cards_snet: cards_snet,
      action_class: ac,
      profile: profile,
      blind_biggest: Money::zero(),
      players: HashSet::new(),
      players_cards: HashMap::new(),
      table_cards: Vec::new(),
    }
  }
}

impl<'a, R: Rng> Players<()> for SNetLex<'a, R> {
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.blind_biggest = *blinds.iter().max().unwrap();
    self.players = (0..self.profile.players).collect();

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
    use anna_eval::strength::return_rate;
    use anna_utils::math;

    for event in events {
      match event {
        &Event::Table { ref cards } => {
          self.table_cards.extend(cards.iter());
        }
        &Event::Play(Act { seat_id, action }) if self.action_class.is_fold(action) => {
          self.players.remove(&seat_id);
        }
        _ => {}
      }
    }

    let hand = self.players_cards.get(&seat_id).unwrap();

    let strength = self
      .cards_snet
      .run(&vec![hand.iter().cloned().collect(), self.table_cards.clone()], self.players.len());

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
