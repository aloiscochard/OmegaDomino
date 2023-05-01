extern crate chrono;
extern crate num_traits;
extern crate rand;

extern crate anna_eval;
extern crate anna_model;

use rand::Rng;

use anna_eval::Eval;
use anna_model::{cards::Card, Action, Money};
use players::{Players, Table};
use Event;
use SeatId;
use Sim;
use State;

// Rules: http://www.homedominotourney.com/docs/rulebook/domino-rule-book-hpt-11-fullsize.pdf

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Error<E> {
  Sementic(String),
  InsufficientBringIn { seat_id: SeatId },
  InsufficientFund { seat_id: SeatId, fund: Money, bet: Money },
  Play(E),
}

impl<E> From<E> for Error<E> {
  fn from(error: E) -> Self {
    Error::Play(error)
  }
}
#[derive(Clone, Debug)]
pub struct Log {
  pub players_init: Vec<(SeatId, Money, Vec<Card>)>,
  pub events: Vec<Event>,
  pub rounds: Rounds,
  pub table_cards: Vec<Card>,
}

pub type Rounds = Vec<Vec<(SeatId, Action, Option<Money>)>>;

#[derive(Clone, Debug)]
pub struct Score {
  pub player_funds: Vec<(SeatId, Money)>,
  pub pot: Money,
  pub winners: Vec<SeatId>,
  pub winners_score: f32,
}

pub fn funds_update(
  sim: &Sim,
  player_funds: &mut Vec<Money>,
  player_pots: &[Money],
  winners: &Vec<SeatId>,
) -> () {
  let prorata = move |player_funds: &mut Vec<Money>, players: &Vec<SeatId>, money: u32| {
    let pot_prorata: Money = Money::from_u32(money / players.len() as u32);
    let pot_remaining: u32 = money - (pot_prorata.unpack() * players.len() as u32);

    if pot_remaining > 0 {
      let mut seat_id: usize = players[0] as usize;
      let mut m = player_funds.iter().max().unwrap().unpack();
      for s in players {
        let pm = player_funds[*s as usize].unpack();
        if pm < m {
          seat_id = *s as usize;
          m = pm;
        }
      }

      player_funds[seat_id] = player_funds[seat_id] + Money::from_u32(pot_remaining);
    }

    for &seat_id in players {
      player_funds[seat_id as usize] = player_funds[seat_id as usize] + pot_prorata;
    }
  };

  let player_pots_max = player_pots.iter().enumerate().map(|(_, m)| m).max().unwrap();

  if winners.iter().any(|&seat_id| player_pots[seat_id as usize] != *player_pots_max) {
    // Compute and distribute side pots.
    let mut xs: Vec<(u32)> =
      winners.iter().map(|&seat_id| player_pots[seat_id as usize].unpack()).collect();

    xs.sort();
    xs.dedup();

    let winners_target = xs.last().unwrap().clone();

    let mut pot_target_last: u32 = 0;

    for pot_target in xs {
      let mut pot_side: u32 = 0;

      let pot_target_delta = pot_target - pot_target_last;

      for seat_id in 0..sim.profile.players {
        let player_pot = player_pots[seat_id].unpack();
        if player_pot > 0 {
          if player_pot >= pot_target {
            pot_side += pot_target_delta;
          } else if player_pot > pot_target_last {
            pot_side += player_pot - pot_target_last;
          }
        }
      }

      let winners_side: Vec<SeatId> = winners
        .iter()
        .filter(|&seat_id| player_pots[*seat_id as usize].unpack() >= pot_target)
        .cloned()
        .collect();

      prorata(player_funds, &winners_side, pot_side);

      pot_target_last = pot_target;
    }

    // If no one did match the last raises while the winner did all-in before them,
    // we'll give money back to those last raises after distributing the winner(s)
    // part.
    if winners_target < player_pots_max.unpack() {
      let xs: Vec<usize> = player_pots
        .iter()
        .enumerate()
        .filter_map(|(seat_id, &m)| if m.unpack() > winners_target { Some(seat_id) } else { None })
        .collect();

      for seat_id in xs {
        let target_max = player_pots[seat_id].unpack();
        player_funds[seat_id] =
          player_funds[seat_id] + Money::from_u32(target_max - winners_target);
      }
    }
  } else {
    // Distribute pot pro rata.
    let pot: u32 = player_pots.iter().map(|m| m.unpack()).sum();
    prorata(player_funds, winners, pot);
  }
}

pub fn game_finish(
  evaluator: &Eval,
  table_cards: &[Card],
  player_states: &[State],
) -> (Vec<(usize, Vec<Card>, bool)>, Vec<SeatId>, f32) {
  let finalists: Vec<(usize, Vec<Card>, bool)> = player_states
    .iter()
    .enumerate()
    .filter_map(|(i, s)| match s {
      &State::Play { ref cards, all_in } => Some((i, cards.clone(), all_in)),
      _ => None,
    })
    .collect();

  let mut winners_score = 0f32;

  let winners = if finalists.len() > 1 {
    // Compute score and elect winner(s)
    let mut winners: Vec<SeatId> = Vec::new();
    for &(seat_id, ref player_cards, _) in finalists.iter() {
      let mut hand = Vec::new();
      hand.extend(table_cards);
      hand.extend(player_cards);

      let score = evaluator.score(&hand);
      if score >= winners_score {
        if score == winners_score {
          winners.push(seat_id);
        } else {
          winners = vec![seat_id];
          winners_score = score;
        }
      }
    }

    winners
  } else {
    let winners: Vec<SeatId> = finalists.iter().map(|&(i, _, _)| i).collect();
    winners
  };

  (finalists, winners, winners_score)
}

pub fn table_simulate<E, R: Rng>(
  rng: &mut R,
  sim: &Sim,
  evaluator: &Eval,
  players: &mut Players<E>,
  player_funds_init: &Vec<(SeatId, Money)>,
  player_first: SeatId,
) -> Result<(Vec<(SeatId, Money)>, Vec<(Log, Score)>), (Vec<(Log, Score)>, Log, Error<E>)> {
  table_simulate_::<E, R>(rng, sim, evaluator, players, player_funds_init, player_first, None)
}

pub fn table_simulate_<E, R: Rng>(
  rng: &mut R,
  sim: &Sim,
  evaluator: &Eval,
  players: &mut Players<E>,
  player_funds_init: &Vec<(SeatId, Money)>,
  player_first: SeatId,
  games_max: Option<usize>,
) -> Result<(Vec<(SeatId, Money)>, Vec<(Log, Score)>), (Vec<(Log, Score)>, Log, Error<E>)> {
  use rounds;
  use std::collections::HashSet;

  let player_funds_sum: u32 = player_funds_init.iter().map(|&(_, m)| m.unpack()).sum::<u32>();

  let ref mut player_funds: Vec<(SeatId, Money)> = player_funds_init.clone();
  let mut actives: HashSet<SeatId> =
    player_funds_init.iter().map(|&(seat_id, _)| seat_id).collect();

  let ref mut scores: Vec<(Log, Score)> = Vec::new();

  let mut deads: usize = 0; // Count of dead game in a row.

  while actives.len() > 1 && deads < 6 && scores.len() < games_max.unwrap_or(usize::max_value()) {
    // Move dealer button
    // TODO Replace naive implementation with official rules!

    let mut seat_id = player_first;

    while {
      seat_id = rounds::seat_next(sim.profile.players, seat_id);
      !actives.contains(&seat_id)
    } {}

    // Extract funds
    let funds: Vec<(SeatId, Money)> =
      player_funds.iter().filter(|&&(seat_id, _)| actives.contains(&seat_id)).map(|&x| x).collect();

    // Play
    let (log, score) = table_game_simulate(rng, sim, evaluator, players, &funds, seat_id)
      .map_err(|(log, err)| (scores.clone(), log, err))?;

    scores.push((log, score.clone()));

    if score.player_funds == funds {
      deads += 1;
    } else {
      deads = 0;

      // Update funds and disable players with insufficient funds.
      player_funds.retain(|&(seat_id, _)| !actives.contains(&seat_id));
      player_funds.extend(score.player_funds);
    }

    // {
    //   player_funds.sort();
    //   println!("{} {:?}", scores.len(), player_funds);
    // }

    // TODO Remove this assertion!
    assert!(player_funds_sum == player_funds.iter().map(|&(_, m)| m.unpack()).sum::<u32>());

    for &(seat_id, money) in player_funds.iter() {
      if (sim.blind_biggest - money) >= 0 {
        actives.remove(&seat_id);
      }
    }
  }

  if deads == 6 {
    let n = scores.len() - 6;
    scores.truncate(n);
  }

  assert!(player_funds_sum == player_funds.iter().map(|&(_, m)| m.unpack()).sum::<u32>());

  Ok((player_funds.clone(), scores.clone()))
}

pub fn table_game_simulate<E, R: Rng>(
  rng: &mut R,
  sim: &Sim,
  evaluator: &Eval,
  players: &mut Players<E>,
  player_funds_init: &Vec<(SeatId, Money)>,
  player_first: SeatId,
) -> Result<(Log, Score), (Log, Error<E>)> {
  use players::TableStatic;

  let ref mut players_init = Vec::new();

  let mut cards: Vec<&Card> = sim.profile.deck.iter().collect();
  rng.shuffle(&mut cards);

  let private_cards_size = sim.profile.rounds[0];

  for &(seat_id, m) in player_funds_init {
    let cs: Vec<&Card> = cards.drain(0..private_cards_size).collect();
    players_init.push((seat_id, m, cs.iter().map(|&c| c.clone()).collect()));
  }

  let table_cards_size = sim.profile.rounds.iter().sum::<usize>() - private_cards_size;
  let ref table_cards: Vec<Card> =
    cards.iter().take(table_cards_size).map(|&c| c.clone()).collect();

  let mut table = TableStatic {
    rounds: sim.profile.rounds.clone(),
    table_cards: table_cards.clone(),
    players: players,
  };

  let (rounds, events, result) =
    table_game_simulate_(sim, evaluator, &mut table, players_init, player_first).map_err(
      |(rounds, events, e)| {
        (
          Log {
            players_init: players_init.clone(),
            events: events,
            rounds: rounds,
            table_cards: table_cards.clone(),
          },
          e,
        )
      },
    )?;

  let score = match result {
    None =>
    // Dead game.
    {
      Score {
        winners: Vec::new(),
        winners_score: 0.,
        player_funds: player_funds_init.clone(),
        pot: Money::zero(),
      }
    }
    Some((mut player_funds, player_pots, player_states)) => {
      let (_, ref winners, winners_score) = game_finish(evaluator, &table_cards, &player_states);
      let pot = Money::from_u32(player_pots.iter().map(|&p| p.unpack()).sum::<u32>());

      funds_update(sim, &mut player_funds, &player_pots, winners);

      // Extract updated funds
      let player_funds_final = players_init
        .iter()
        .map(|&(seat_id, _, _)| (seat_id, player_funds[seat_id as usize]))
        .collect();

      Score {
        winners: winners.clone(),
        winners_score: winners_score,
        player_funds: player_funds_final,
        pot: pot,
      }
    }
  };

  Ok((
    Log {
      players_init: players_init.clone(),
      events: events,
      rounds: rounds,
      table_cards: table_cards.clone(),
    },
    score,
  ))
}

pub fn table_game_simulate_<E, PT>(
  sim: &Sim,
  evaluator: &Eval,
  players_table: &mut PT,
  players_init: &Vec<(SeatId, Money, Vec<Card>)>,
  player_first: SeatId,
) -> Result<
  (Rounds, Vec<Event>, Option<(Vec<Money>, Vec<Money>, Vec<State>)>),
  (Rounds, Vec<Event>, Error<E>),
>
where
  PT: Players<E>,
  PT: Table<E>,
{
  use anna_model::profile::Limit;
  use rounds;
  use std::cmp::min;
  use Act;
  use Event;

  let blind_biggest = sim.blind_biggest;

  // TODO Check also that all *active* users have a blind to set `ante`.
  let blinds_ante: bool = sim.profile.blinds.iter().all(|&m| m == blind_biggest);

  let dealer = rounds::seat_previous(sim.profile.players, player_first);

  let game_abort = |rounds: &Vec<Vec<(usize, Action, Option<Money>)>>,
                    rounds_buffer: &Vec<(usize, Action, Option<Money>)>| {
    let mut xs = rounds.clone();
    xs.push(rounds_buffer.clone());
    xs
  };

  let round_start =
    move |evaluator: &Eval, table: &mut Table<E>, round_id: usize, table_target: Money| {
      let table_cards = table.round_start(evaluator, round_id, table_target)?;
      Ok(Event::Table { cards: table_cards })
    };

  let mut running = true;

  let mut table_bootstrapped = false;

  let (mut player_funds, mut player_pots, mut player_states) =
    table_players_init(sim, players_init).map_err(|e| (Vec::new(), Vec::new(), e))?;

  let mut table_target: Money = blind_biggest; // Target for the current bet
  let mut table_target_raise: Money = blind_biggest; // - amount by which the target was raised

  let ref mut events: Vec<Event> = Vec::new();

  let mut round_raises: usize = 0; // Count of raises for the current round
  let mut rounds: Vec<Vec<(usize, Action, Option<Money>)>> = Vec::new();
  let ref mut rounds_buffer: Vec<(usize, Action, Option<Money>)> = Vec::new();

  // Initialize game.
  let player_cards =
    players_init.iter().map(|&(seat_id, _, ref cards)| (seat_id as usize, cards.clone())).collect();
  players_table.init(&sim.profile.blinds, player_first, &player_funds, &player_cards);

  let (mut active, blinds_actions) =
    rounds::blinds_apply(sim, player_first, &mut player_funds, &mut player_pots, &player_states);

  let blinds_player_last = blinds_actions.iter().last().unwrap().0.seat_id;

  players_table
    .game_start(blinds_actions.iter().map(|&(ref act, m)| (act.seat_id, m)).collect())
    .map_err(|e| (Vec::new(), Vec::new(), Error::Play(e)))?;

  let _ = round_start(evaluator, players_table, 0, table_target)
    .map_err(|e| (Vec::new(), Vec::new(), Error::Play(e)))?;

  for &(ref act, m) in blinds_actions.iter() {
    rounds_buffer.push((act.seat_id, Action::Raise, Some(m)));
    if !blinds_ante {
      events.push(Event::Play(act.clone()));
    }
  }

  // Run the game ...
  let mut round_id: usize = 0;
  let mut action_id: usize = blinds_actions.len(); // Idx of action in a round

  // Who set the current target to match
  let mut table_target_by: Option<usize> = if blinds_ante { Some(active) } else { None };

  while running {
    if action_id == 0 {
      // Initialize rounds
      let ev_table = round_start(evaluator, players_table, round_id, table_target)
        .map_err(|e| (game_abort(&rounds, &rounds_buffer), events.clone(), Error::Play(e)))?;
      events.push(ev_table);
    }

    // Extract events until last active player act.
    let events_player: Vec<Event> = {
      let events_begin = if !table_bootstrapped {
        if active == blinds_player_last {
          table_bootstrapped = true
        };

        0
      } else {
        let takes = events
          .iter()
          .rev()
          .take_while(|&a| match a {
            &Event::Play(Act { seat_id, .. }) => seat_id != active,
            &Event::Table { .. } => true,
          })
          .count();

        events.len() - (takes + 1)
      };

      events[events_begin..].to_vec()
    };

    // Play
    let action_raisable = match sim.profile.limit {
      None => true,
      Some(Limit { caps, .. }) => round_raises < caps,
    };

    let action_i = players_table
      .play(
        round_id,
        table_target,
        if action_raisable { Some(table_target_raise) } else { None },
        &player_pots,
        active,
        player_funds[active],
        &events_player,
      )
      .map_err(|e| (game_abort(&rounds, &rounds_buffer), events.clone(), Error::Play(e)))?;

    // Apply action and record event
    let (action, money_opt, _) = table_run(
      sim,
      round_id,
      &mut player_funds,
      &mut player_pots,
      &mut player_states,
      &mut table_target,
      &mut table_target_raise,
      &mut table_target_by,
      active,
      action_i,
    )
    .map_err(|e| (game_abort(&rounds, &rounds_buffer), events.clone(), e))?;

    let event = Event::Play(Act { seat_id: active, action: action_i });

    events.push(event);

    rounds_buffer.push((active, action, money_opt));

    if action == Action::Raise {
      assert!(action_raisable);
      round_raises += 1
    }

    action_id += 1;

    match table_next(
      sim,
      &player_pots,
      &player_states,
      &table_target,
      &mut table_target_raise,
      &mut table_target_by,
      dealer,
      round_id,
      active,
    ) {
      Err(None) => {
        running = false;
      }
      Err(Some(next)) => {
        active = next;

        rounds.push(rounds_buffer.clone());
        rounds_buffer.clear();

        round_id += 1;
        round_raises = 0;
        action_id = 0;
      }
      Ok(next) => {
        active = next;
      }
    }
  }

  rounds.push(rounds_buffer.clone());

  Ok(if table_target_by.is_none() {
    // Dead game, no call/raise was made.
    for &(ref act, m) in blinds_actions.iter() {
      rounds::pledge_unapply(&mut player_funds, &mut player_pots, act.seat_id, m);
    }
    (rounds, events.clone(), None)
  } else {
    (rounds, events.clone(), Some((player_funds, player_pots, player_states)))
  })
}

pub fn table_next(
  sim: &Sim,
  player_pots: &[Money],
  player_states: &[State],
  table_target: &Money,
  table_target_raise: &mut Money,
  table_target_by: &mut Option<usize>,
  dealer: usize,
  round_id: usize,
  seat_id: usize,
) -> Result<usize, Option<usize>> {
  use rounds::{player_actives, seat_next_active};

  let player_actives: Vec<usize> = player_actives(&player_states);

  if player_actives.len() == 0
    || (player_actives.len() == 1 && (*table_target - player_pots[player_actives[0]]) <= 0)
  {
    Err(None)
  } else {
    // Next action
    let seat_id_next = seat_next_active(sim.profile.players, &player_actives, seat_id).unwrap();
    if Some(seat_id_next) == *table_target_by {
      assert!(player_pots[seat_id_next] == *table_target);

      // We reached the end of the round, no more betting was done.
      if round_id == (sim.profile.rounds.len() - 1) {
        Err(None)
      } else {
        let seat_id_init = seat_next_active(sim.profile.players, &player_actives, dealer).unwrap();

        *table_target_raise = sim.blind_biggest;
        *table_target_by = Some(seat_id_init);

        Err(Some(seat_id_init))
      }
    } else {
      Ok(seat_id_next)
    }
  }
}

pub fn table_players_init<E>(
  sim: &Sim,
  players_init: &Vec<(SeatId, Money, Vec<Card>)>,
) -> Result<(Vec<Money>, Vec<Money>, Vec<State>), Error<E>> {
  let mut player_funds: Vec<Money> = vec![Money::zero(); sim.profile.players];
  let player_pots: Vec<Money> = vec![Money::zero(); sim.profile.players];
  let mut player_states: Vec<State> = Vec::new();
  for _ in 0..sim.profile.players {
    player_states.push(State::Off);
  }

  for &(seat_id, money, ref cards) in players_init.iter() {
    let idx = seat_id as usize;
    player_funds[idx] = money;
    player_states[idx] = State::Play { cards: cards.clone(), all_in: false };

    if (sim.blind_biggest - money) > 0 {
      return Err(Error::InsufficientBringIn { seat_id: seat_id });
    }
  }

  Ok((player_funds, player_pots, player_states))
}

pub fn table_run<E>(
  sim: &Sim,
  round_id: usize,
  player_funds: &mut Vec<Money>,
  player_pots: &mut Vec<Money>,
  player_states: &mut Vec<State>,
  table_target: &mut Money,
  table_target_raise: &mut Money,
  table_target_by: &mut Option<usize>,
  seat_id: SeatId,
  action_i: u8,
) -> Result<(Action, Option<Money>, Option<Money>), Error<E>> {
  use rounds;

  let (action, money_opt) = sim
    .action_class
    .unapply(
      sim.blind_biggest,
      round_id,
      player_funds[seat_id],
      *table_target,
      *table_target_raise,
      player_pots[seat_id],
      action_i,
    )
    .map_err(Error::Sementic)?;

  let mut money_norm = money_opt.clone();

  // println!("{} {} {} - {:?} {:?} / {:?}", round_id, seat_id, action_i, action,
  // money_opt, player_funds[seat_id]);

  if action == Action::Fold {
    // Fold
    if player_pots[seat_id] >= *table_target && sim.strict {
      return Err(Error::Sementic(format!(
        "Can not `Fold` when already at target. player {}",
        seat_id
      )));
    } else {
      player_states[seat_id] = State::Folded;
    }
  } else if action == Action::Call && money_opt.is_none() {
    // Check
    if player_pots[seat_id] != *table_target {
      return Err(Error::Sementic(format!(
        "Can not `Check` when target is not reached. player {}",
        seat_id
      )));
    }
  } else {
    // Call/Raise
    let money = money_opt.expect("Impossible to Call/Raise without money.");

    if action == Action::Call && (*table_target - player_pots[seat_id]) <= 0 {
      return Err(Error::Sementic(format!(
        "Can not `Call` when already at target. player {}",
        seat_id
      )));
    }

    let all_in = money == player_funds[seat_id] || (!sim.strict && (money > player_funds[seat_id]));
    if all_in {
      // AllIn
      let state = match player_states[seat_id] {
        State::Play { ref cards, .. } => State::Play { cards: cards.clone(), all_in: true },
        _ => {
          panic!("Can not `AllIn` on a non active player {}", seat_id);
        }
      };
      player_states[seat_id] = state;
    }

    if !rounds::pledge_apply(player_funds, player_pots, seat_id, money) {
      return Err(Error::InsufficientFund {
        seat_id: seat_id,
        fund: player_funds[seat_id],
        bet: money,
      });
    }

    let raise = player_pots[seat_id] - *table_target;
    if action == Action::Call {
      if all_in {
        // Apply "Full Bet" rule during AllIns.
        // http://www.dominolistings.com/rules-for-domino-all-in-situations-domino-side-pot-calculator
        if raise > 0 {
          assert!((raise as u32) < table_target_raise.unpack());
          money_norm = Some(Money::from_u32(money.unpack() - raise as u32));
          rounds::pledge_unapply(
            player_funds,
            player_pots,
            seat_id,
            Money::from_i32(raise).unwrap(),
          );
        }
      } else {
        assert!(raise == 0);
      }

      let target_set = match *table_target_by {
        Some(table_target_by) => match player_states[table_target_by] {
          State::Play { all_in, .. } => all_in,
          _ => {
            panic!("Can not have a non active player ({}) as table_target.", table_target_by);
          }
        },
        None => true,
      };

      if target_set {
        *table_target_by = Some(seat_id);
      };
    } else {
      assert!(raise > 0 && raise as u32 >= table_target_raise.unpack());

      *table_target_raise = Money::from_i32(raise).unwrap();
      *table_target = player_pots[seat_id];
      *table_target_by = Some(seat_id);
    }
  }

  Ok((action, money_opt, money_norm))
}
