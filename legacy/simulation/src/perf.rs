use anna_eval::Eval;
use anna_model::{cards::Card, Money};
use players::Players;
use rand::Rng;
use std::fmt::Debug;
use SeatId;
use Sim;

// Win rate - Milli big blind per hand.
pub type Rate = f32;

pub fn rate(blind_biggest: Money, fs_init: Money, fs_final: Money) -> Rate {
  (1000.0 * ((fs_final - fs_init) as f32 / blind_biggest.unpack() as f32))
}

pub fn rate_format(r: Rate) -> String {
  format!("{:.2}", r)
}

pub fn rate_players(
  blind_biggest: Money,
  players_init: &Vec<(SeatId, Money, Vec<Card>)>,
  player_funds: &Vec<(SeatId, Money)>,
) -> Vec<(SeatId, Rate)> {
  let mut rates = Vec::new();

  for &(seat_id, m_s, _) in players_init {
    let &(_, m_t) = player_funds.iter().find(|&&(s, _)| seat_id == s).unwrap();
    rates.push((seat_id, rate(blind_biggest, m_s, m_t)));
  }

  rates
}

pub fn run_benchmark<E: Debug, P: Players<E>>(
  sim: &Sim,
  evaluator: &Eval,
  player_funds: &Vec<(SeatId, Money)>,
  hands: usize,
  players: &mut P,
) -> Vec<Rate> {
  use anna_utils::math;
  use engine::table_game_simulate;
  use rand::thread_rng;
  use rayon::{current_num_threads, prelude::*};

  // let threads = current_num_threads();
  // let hands_per_worker = hands / threads;

  let hands_per_worker = hands / 1;

  let ref mut rng = thread_rng();

  let mut seats: Vec<SeatId> = player_funds.iter().map(|&(seat_id, _)| seat_id).collect();
  let mut players_rates = vec![Vec::new(); sim.profile.players];
  let mut n = 0;

  while n < hands_per_worker {
    let player_first = seats[n % seats.len()];

    let (log, score) =
      table_game_simulate(rng, sim, evaluator, players, player_funds, player_first)
        .map_err(|(_, err)| err)
        .expect("simulation failed.");
    let rates = rate_players(sim.blind_biggest, &log.players_init, &score.player_funds);

    for (seat_id, rate) in rates {
      players_rates[seat_id].push(rate);
    }

    n += 1;
  }

  // let mut players_rates = vec![Vec::new(); sim.profile.players];
  // for xss in results {
  //   for (seat_id, xs) in xss.iter().enumerate() {
  //     players_rates[seat_id].extend(xs);
  //   }
  // }

  players_rates.iter().map(math::mean).collect()
}
