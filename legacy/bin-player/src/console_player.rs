extern crate anna_eval;
extern crate anna_model;
extern crate anna_simulation;

use anna_eval::Eval;
use anna_model::{cards::Card, money::Money};
use anna_simulation::{
  players::{Players, Table},
  Event,
};

use ui;

pub struct PlayerUI<
  'a,
  P: Players<()>,
  GS: FnMut(&mut StateUI<P>, Vec<(usize, Money)>) -> Result<(), ui::Exit>,
  RS: FnMut(&mut StateUI<P>, usize, Money) -> Result<Vec<Card>, ui::Exit>,
  IN: FnMut(&mut StateUI<P>, &[Money], usize, &[Money], &Vec<(usize, Vec<Card>)>) -> (),
  PL: FnMut(
    &mut StateUI<P>,
    usize,
    Money,
    Option<Money>,
    &[Money],
    usize,
    Money,
    &[Event],
  ) -> Result<u8, ui::Exit>,
> {
  pub game_start: GS,
  pub round_start: RS,
  pub init: IN,
  pub play: PL,
  pub state: StateUI<'a, P>,
}

pub struct StateUI<'a, P: Players<()>> {
  pub players: &'a mut P,
  pub player_funds: Vec<Money>,
  pub player_pots: Vec<Money>,
  pub table_cards: Vec<Card>,
  pub table_target_init: Money, // Table target at start of the current round.
}

impl<'a, P: Players<()>, GS, RS, IN, PL> Players<ui::Exit> for PlayerUI<'a, P, GS, RS, IN, PL>
where
  GS: FnMut(&mut StateUI<P>, Vec<(usize, Money)>) -> Result<(), ui::Exit>,
  RS: FnMut(&mut StateUI<P>, usize, Money) -> Result<Vec<Card>, ui::Exit>,
  IN: FnMut(&mut StateUI<P>, &[Money], usize, &[Money], &Vec<(usize, Vec<Card>)>) -> (),
  PL: FnMut(
    &mut StateUI<P>,
    usize,
    Money,
    Option<Money>,
    &[Money],
    usize,
    Money,
    &[Event],
  ) -> Result<u8, ui::Exit>,
{
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: usize,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    (self.init)(&mut self.state, blinds, player_first, player_funds, playing_hands)
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
  ) -> Result<u8, ui::Exit> {
    (self.play)(
      &mut self.state,
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

impl<'a, P: Players<()>, GS, RS, IN, PL> Table<ui::Exit> for PlayerUI<'a, P, GS, RS, IN, PL>
where
  GS: FnMut(&mut StateUI<P>, Vec<(usize, Money)>) -> Result<(), ui::Exit>,
  RS: FnMut(&mut StateUI<P>, usize, Money) -> Result<Vec<Card>, ui::Exit>,
  IN: FnMut(&mut StateUI<P>, &[Money], usize, &[Money], &Vec<(usize, Vec<Card>)>) -> (),
  PL: FnMut(
    &mut StateUI<P>,
    usize,
    Money,
    Option<Money>,
    &[Money],
    usize,
    Money,
    &[Event],
  ) -> Result<u8, ui::Exit>,
{
  fn game_start(&mut self, blinds: Vec<(usize, Money)>) -> Result<(), ui::Exit> {
    (self.game_start)(&mut self.state, blinds)
  }
  fn round_start(
    &mut self,
    _: &Eval,
    round_id: usize,
    table_target: Money,
  ) -> Result<Vec<Card>, ui::Exit> {
    (self.round_start)(&mut self.state, round_id, table_target)
  }
}
