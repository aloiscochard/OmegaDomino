extern crate anna_model;

use anna_model::{Action, Money};
use Act;
use SeatId;
use Sim;
use State;

pub fn action_apply(
  sim: &Sim,
  player_funds: &mut Vec<Money>,
  player_pots: &mut Vec<Money>,
  seat_id: SeatId,
  action: Action,
  money: Option<Money>,
) -> u8 {
  for money in money {
    assert!(pledge_apply(player_funds, player_pots, seat_id, money));
  }
  sim.action_class.apply(
    sim.blind_biggest,
    player_funds[seat_id as usize],
    action,
    money.unwrap_or(Money::zero()),
  )
}

pub fn blinds_apply(
  sim: &Sim,
  player_first: SeatId,
  player_funds: &mut Vec<Money>,
  player_pots: &mut Vec<Money>,
  player_states: &Vec<State>,
) -> (SeatId, Vec<(Act, Money)>) {
  let player_actives: Vec<SeatId> = player_actives(&player_states);

  let mut active = player_first;
  let mut raises = Vec::new();

  for &blind in sim.profile.blinds.iter().take(player_actives.len()) {
    let action_i = action_apply(sim, player_funds, player_pots, active, Action::Raise, Some(blind));
    raises.push((Act { seat_id: active, action: action_i }, blind));
    active = seat_next_active(sim.profile.players, &player_actives, active).unwrap();
  }

  (active, raises)
}

pub fn player_actives(player_states: &[State]) -> Vec<SeatId> {
  player_states
    .iter()
    .enumerate()
    .filter_map(|(i, s)| match s {
      &State::Play { all_in: false, .. } => Some(i),
      _ => None,
    })
    .collect()
}

pub fn pledge_apply(
  player_funds: &mut Vec<Money>,
  player_pots: &mut Vec<Money>,
  seat_id: usize,
  money: Money,
) -> bool {
  let delta = player_funds[seat_id] - money;
  if delta >= 0 {
    player_funds[seat_id] = Money::from_i32(delta).unwrap();
    player_pots[seat_id] = player_pots[seat_id] + money;
    true
  } else {
    false
  }
}

pub fn pledge_unapply(
  player_funds: &mut Vec<Money>,
  player_pots: &mut Vec<Money>,
  seat_id: usize,
  money: Money,
) -> () {
  player_pots[seat_id] = Money::from_i32(player_pots[seat_id] - money).unwrap();
  player_funds[seat_id] = player_funds[seat_id] + money;
}

pub fn seat_next(players: usize, seat_id: SeatId) -> SeatId {
  if seat_id == (players - 1) {
    0
  } else {
    seat_id + 1
  }
}

pub fn seat_next_active(
  players: usize,
  player_actives: &[SeatId],
  seat_id: SeatId,
) -> Option<SeatId> {
  let mut value = seat_next(players, seat_id);
  for _ in 0..players {
    if !player_actives.contains(&value) {
      value = seat_next(players, value);
    } else {
      return Some(value);
    }
  }
  None
}

pub fn seat_previous(players: usize, seat_id: SeatId) -> SeatId {
  if seat_id == 0 {
    players - 1
  } else {
    seat_id - 1
  }
}
