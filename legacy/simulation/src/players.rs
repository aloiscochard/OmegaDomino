use std::collections::HashSet;

use rand;

use anna_eval::Eval;
use anna_model::{cards::Card, ActionClass, Money};

use Event;
use SeatId;

pub trait Players<E> {
  /** Reset the players internal state in-between games on a given table. *
   ** * * * * * * * * * * * * * * * * **/
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> ();
  fn play(
    &mut self,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_pots: &[Money],
    seat_id: usize,
    player_fund: Money,
    events: &[Event],
  ) -> Result<u8, E>;
}

#[derive(Clone)]
pub struct PlayersFold<'a> {
  action_class: &'a ActionClass,
  blind_biggest: Money,
}

impl<'a> PlayersFold<'a> {
  pub fn new(action_class: &'a ActionClass) -> PlayersFold {
    PlayersFold { action_class: action_class, blind_biggest: Money::zero() }
  }
}

impl<'a> Players<()> for PlayersFold<'a> {
  fn init(&mut self, blinds: &[Money], _: SeatId, _: &[Money], _: &Vec<(usize, Vec<Card>)>) -> () {
    self.blind_biggest = *blinds.iter().max().unwrap();
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
    let mask = self.action_class.normalize(
      self.blind_biggest,
      round_id,
      table_target,
      table_target_raise,
      player_fund,
      player_pots[seat_id],
    );

    for i in 0..self.action_class.size() {
      if mask.contains(&i) {
        return Ok(i as u8);
      }
    }
    Err(())
  }
}

#[derive(Clone)]
pub struct PlayersRand<'a, R: rand::Rng> {
  pub action_class: &'a ActionClass,
  pub blind_biggest: Money,
  pub rng: R,
}

impl<'a, R: rand::Rng> PlayersRand<'a, R> {
  pub fn new(rng: R, action_class: &'a ActionClass, blind_biggest: Money) -> PlayersRand<'a, R> {
    PlayersRand { action_class: action_class, blind_biggest: blind_biggest, rng: rng }
  }
}

impl<'a, R: rand::Rng> Players<()> for PlayersRand<'a, R> {
  fn init(&mut self, blinds: &[Money], _: SeatId, _: &[Money], _: &Vec<(usize, Vec<Card>)>) -> () {
    self.blind_biggest = *blinds.iter().max().unwrap();
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
    use anna_utils::math;

    let mask = self.action_class.normalize(
      self.blind_biggest,
      round_id,
      table_target,
      table_target_raise,
      player_fund,
      player_pots[seat_id],
    );

    let probs =
      (0..self.action_class.size()).map(|i| if mask.contains(&i) { 1.0 } else { 0.0 }).collect();
    let action = math::sample(&mut self.rng, probs);
    Ok(action as u8)
  }
}

pub struct PlayersStatic(Vec<(SeatId, u8)>);

impl Players<()> for PlayersStatic {
  fn init(&mut self, _: &[Money], _: SeatId, _: &[Money], _: &Vec<(usize, Vec<Card>)>) -> () {}
  fn play(
    &mut self,
    _: usize,
    _: Money,
    _: Option<Money>,
    _: &[Money],
    seat_id: usize,
    _: Money,
    _: &[Event],
  ) -> Result<u8, ()> {
    let (seat, action) = self.0.remove(0);
    if seat == seat_id {
      Ok(action)
    } else {
      Err(())
    }
  }
}

pub trait Table<E> {
  fn game_start(&mut self, blinds: Vec<(usize, Money)>) -> Result<(), E>;
  fn round_start(
    &mut self,
    evaluator: &Eval,
    round_id: usize,
    table_target: Money,
  ) -> Result<Vec<Card>, E>;
}

pub struct TableStatic<'a, E: 'a> {
  pub rounds: Vec<usize>,
  pub table_cards: Vec<Card>,
  pub players: &'a mut Players<E>,
}

impl<'a, E> Table<E> for TableStatic<'a, E> {
  fn game_start(&mut self, _: Vec<(usize, Money)>) -> Result<(), E> {
    Ok(())
  }

  fn round_start(&mut self, _: &Eval, round_id: usize, _: Money) -> Result<Vec<Card>, E> {
    let mut skips = 0;
    let mut takes = 0;

    for i in 1..(round_id + 1) {
      if round_id == i {
        takes = self.rounds[i];
      } else {
        skips += self.rounds[i];
      }
    }

    Ok(self.table_cards.iter().skip(skips).take(takes).map(|&c| c).collect())
  }
}

impl<'a, E> Players<E> for TableStatic<'a, E> {
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) {
    self.players.init(blinds, player_first, player_funds, playing_hands)
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
  ) -> Result<u8, E> {
    self.players.play(
      round_id,
      table_target,
      table_target_raise,
      player_pots,
      seat_id,
      player_fund,
      events,
    )
  }
}
