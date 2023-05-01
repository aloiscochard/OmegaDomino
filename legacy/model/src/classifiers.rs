use cards::Card;
use std::collections::HashSet;
use Action;
use ActionClass;
use Money;

#[derive(Clone)]
pub struct ActionKuhn {}

impl ActionClass for ActionKuhn {
  fn apply(&self, _: Money, _: Money, action: Action, _: Money) -> u8 {
    action as u8
  }

  fn unapply(
    &self,
    blind_biggest: Money,
    _: usize,
    _: Money,
    table_target: Money,
    _: Money,
    player_pot: Money,
    action_class: u8,
  ) -> Result<(Action, Option<Money>), String> {
    match action_class {
      0 => Ok((Action::Fold, None)),
      1 => Ok((Action::Call, if player_pot == table_target { None } else { Some(blind_biggest) })),
      2 => Ok((
        Action::Raise,
        Some(Money::from_i32((table_target + blind_biggest) - player_pot).unwrap()),
      )),
      i => panic!(format!("Invalid action: {}", i)),
    }
  }

  fn is_fold(&self, class: u8) -> bool {
    class == 0
  }
  fn is_raise(&self, class: u8) -> bool {
    class == 2
  }

  fn normalize(
    &self,
    blind_biggest: Money,
    _: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_fund: Money,
    player_pot: Money,
  ) -> HashSet<usize> {
    let raisable = table_target_raise.is_some() && player_fund >= blind_biggest;
    let callable = if player_pot >= table_target { true } else { player_fund >= blind_biggest };

    let mut xs = HashSet::new();
    if player_pot < table_target {
      xs.insert(0);
    }
    if callable {
      xs.insert(1);
    }
    if raisable {
      xs.insert(2);
    }

    xs
  }

  fn size(&self) -> usize {
    3
  }
}

#[derive(Clone)]
pub struct ActionLimit {
  pub raises: Vec<usize>, // As ratio of big blind
}

impl ActionClass for ActionLimit {
  fn apply(&self, _: Money, _: Money, action: Action, _: Money) -> u8 {
    action as u8
  }

  fn unapply(
    &self,
    blind_biggest: Money,
    round_id: usize,
    _: Money,
    table_target: Money,
    _: Money,
    player_pot: Money,
    action_class: u8,
  ) -> Result<(Action, Option<Money>), String> {
    match action_class {
      0 => Ok((Action::Fold, None)),
      1 => {
        let money = if player_pot == table_target {
          None
        } else {
          Some(Money::from_i32(table_target - player_pot).unwrap())
        };
        Ok((Action::Call, money))
      }
      2 => {
        let raise = blind_biggest * (self.raises[round_id] as u32);
        Ok((Action::Raise, Some(Money::from_i32((table_target + raise) - player_pot).unwrap())))
      }
      i => panic!(format!("Invalid action: {}", i)),
    }
  }

  fn is_fold(&self, class: u8) -> bool {
    class == 0
  }
  fn is_raise(&self, class: u8) -> bool {
    class == 2
  }

  fn normalize(
    &self,
    blind_biggest: Money,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_fund: Money,
    player_pot: Money,
  ) -> HashSet<usize> {
    let raise = blind_biggest * (self.raises[round_id] as u32);
    let raisable = table_target_raise.is_some() && player_fund >= raise;
    let callable = if player_pot >= table_target {
      true
    } else {
      ((player_fund + player_pot) - table_target) > 0
    };

    let mut xs = HashSet::new();
    if player_pot < table_target {
      xs.insert(0);
    }
    if callable {
      xs.insert(1);
    }
    if raisable {
      xs.insert(2);
    }

    xs
  }

  fn size(&self) -> usize {
    3
  }
}
