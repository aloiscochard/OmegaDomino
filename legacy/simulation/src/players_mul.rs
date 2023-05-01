use std::collections::HashSet;

use anna_eval::Eval;
use anna_model::{cards::Card, ActionClass, Money};

use players::*;

use Event;
use SeatId;

/** Players Multiplexer **/
pub trait PlayersMul<E> {
  fn select<'a>(&'a mut self, seat_id: SeatId) -> &'a mut Players<E>;
  fn select_all<'a>(&'a mut self) -> Vec<(&'a mut Players<E>, &HashSet<usize>)>;

  fn mul_init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    let mut players = self.select_all();

    let mut entries: Vec<Vec<(usize, Vec<Card>)>> = players.iter().map(|_| Vec::new()).collect();

    for &(seat_id, ref cards) in playing_hands {
      let mut found = false;
      for (i, &(_, ref mask)) in players.iter().enumerate() {
        if mask.contains(&seat_id) {
          assert!(!found);
          entries[i].push((seat_id, cards.clone()));
          found = true;
        }
      }
      assert!(found);
    }

    for (i, players_cards) in entries.iter().enumerate() {
      players[i].0.init(blinds, player_first, player_funds, players_cards);
    }
  }

  fn mul_play(
    &mut self,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_pots: &[Money],
    seat_id: usize,
    player_fund: Money,
    events: &[Event],
  ) -> Result<u8, E> {
    self.select(seat_id).play(
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

#[derive(Clone)]
pub struct PlayersMulX<P>(pub Vec<(P, HashSet<usize>)>);

impl<P> PlayersMulX<P> {
  pub fn at(&mut self, i: usize) -> &mut P {
    &mut self.0[i].0
  }

  pub fn mask(&self) -> HashSet<usize> {
    let mut xs = HashSet::new();
    for &(_, ref m) in self.0.iter() {
      xs.extend(m.iter());
    }
    xs
  }

  pub fn mask_set(&mut self, i: usize, mask: HashSet<usize>) -> () {
    self.0[i].1 = mask;
  }
}

impl<P, E> PlayersMul<E> for PlayersMulX<P>
where
  P: Players<E>,
{
  fn select<'a>(&'a mut self, seat_id: SeatId) -> &'a mut Players<E> {
    for &mut (ref mut p, ref mask) in self.0.iter_mut() {
      if mask.contains(&seat_id) {
        return p;
      }
    }
    panic!("No Players for seat {}", seat_id);
  }

  fn select_all<'a>(&'a mut self) -> Vec<(&'a mut Players<E>, &HashSet<usize>)> {
    self.0.iter_mut().map(|&mut (ref mut p, ref mask)| (p as &mut Players<E>, mask)).collect()
  }
}

impl<P, E> Players<E> for PlayersMulX<P>
where
  P: Players<E>,
{
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.mul_init(blinds, player_first, player_funds, playing_hands)
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
    self.mul_play(
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

#[derive(Clone)]
pub struct PlayersMul2<P0, P1>(pub (P0, HashSet<usize>), pub (P1, HashSet<usize>));

impl<P0, P1, E> PlayersMul<E> for PlayersMul2<P0, P1>
where
  P0: Players<E>,
  P1: Players<E>,
{
  fn select<'a>(&'a mut self, seat_id: SeatId) -> &'a mut Players<E> {
    if (self.0).1.contains(&seat_id) {
      return &mut (self.0).0;
    }
    if (self.1).1.contains(&seat_id) {
      return &mut (self.1).0;
    }
    panic!("No Players for seat {}", seat_id);
  }

  fn select_all<'a>(&'a mut self) -> Vec<(&'a mut Players<E>, &HashSet<usize>)> {
    vec![(&mut (self.0).0, &(self.0).1), (&mut (self.1).0, &(self.1).1)]
  }
}

impl<P0, P1, E> Players<E> for PlayersMul2<P0, P1>
where
  P0: Players<E>,
  P1: Players<E>,
{
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.mul_init(blinds, player_first, player_funds, playing_hands)
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
    self.mul_play(
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

#[derive(Clone)]
pub struct PlayersMul3<P0, P1, P2>(
  pub (P0, HashSet<usize>),
  pub (P1, HashSet<usize>),
  pub (P2, HashSet<usize>),
);

impl<P0, P1, P2, E> PlayersMul<E> for PlayersMul3<P0, P1, P2>
where
  P0: Players<E>,
  P1: Players<E>,
  P2: Players<E>,
{
  fn select<'a>(&'a mut self, seat_id: SeatId) -> &'a mut Players<E> {
    if (self.0).1.contains(&seat_id) {
      return &mut (self.0).0;
    }
    if (self.1).1.contains(&seat_id) {
      return &mut (self.1).0;
    }
    if (self.2).1.contains(&seat_id) {
      return &mut (self.2).0;
    }
    panic!("No Players for seat {}", seat_id);
  }

  fn select_all<'a>(&'a mut self) -> Vec<(&'a mut Players<E>, &HashSet<usize>)> {
    vec![
      (&mut (self.0).0, &(self.0).1),
      (&mut (self.1).0, &(self.1).1),
      (&mut (self.2).0, &(self.2).1),
    ]
  }
}

impl<P0, P1, P2, E> Players<E> for PlayersMul3<P0, P1, P2>
where
  P0: Players<E>,
  P1: Players<E>,
  P2: Players<E>,
{
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: SeatId,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.mul_init(blinds, player_first, player_funds, playing_hands)
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
    self.mul_play(
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
