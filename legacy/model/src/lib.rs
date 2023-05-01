extern crate serde;
#[macro_use]
extern crate serde_derive;

pub mod cards;
pub mod classifiers;
pub mod encoders;
pub mod hparam;
pub mod money;
pub mod profile;

pub use money::Money;

use std::collections::HashSet;

pub type SeatId = usize;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Action {
  Fold,  // or Muck
  Call,  // or Check
  Raise, // or Bet
}

pub trait ActionClass: Sync {
  fn apply(&self, blind_biggest: Money, player_fund: Money, action: Action, pledge: Money) -> u8;

  fn unapply(
    &self,
    blind_biggest: Money,
    round_id: usize,
    player_fund: Money,
    table_target: Money,
    table_target_raise: Money,
    player_pot: Money,
    action_class: u8,
  ) -> Result<(Action, Option<Money>), String>;

  fn is_fold(&self, class: u8) -> bool;
  fn is_raise(&self, class: u8) -> bool;

  // TODO Rename to 'mask' ?
  fn normalize(
    &self,
    blind_biggest: Money,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_fund: Money,
    player_pot: Money,
  ) -> HashSet<usize>;

  fn size(&self) -> usize;

  fn to_action(&self, class: u8) -> Action {
    if self.is_fold(class) {
      Action::Fold
    } else if self.is_raise(class) {
      Action::Raise
    } else {
      Action::Call
    }
  }
}
