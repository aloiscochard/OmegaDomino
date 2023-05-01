#[macro_use]
extern crate log;
extern crate rand;
extern crate rayon;

extern crate anna_eval;
extern crate anna_model;
extern crate anna_utils;

pub mod engine;
pub mod perf;
pub mod players;
pub mod players_kuhn;
pub mod players_lex;
pub mod players_mul;
pub mod players_sel;
pub mod rounds;

use anna_model::{cards::Card, profile::Profile, ActionClass, Money};

#[derive(Clone, Debug)]
pub struct Act {
  pub seat_id: usize,
  pub action: u8,
}

#[derive(Clone, Debug)]
pub enum Event {
  Play(Act),
  Table { cards: Vec<Card> },
}

pub type SeatId = usize;

#[derive(Clone)]
pub struct Sim<'a> {
  pub action_class: &'a ActionClass,
  pub blind_biggest: Money,
  pub profile: Profile,
  // stric mode enabled/disabled,
  //   when disabled, tolerate
  //   - folding when at target
  //   - all-in which don't strictly match the bet
  pub strict: bool,
}

impl<'a> Sim<'a> {
  pub fn set_blinds(&mut self, blinds: &[Money]) -> () {
    self.blind_biggest = blinds.iter().max().cloned().expect("Can't set empty blinds.");
    self.profile.blinds = blinds.to_vec();
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum State {
  Play { cards: Vec<Card>, all_in: bool },
  Folded,
  Off,
}
