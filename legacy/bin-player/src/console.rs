extern crate ncurses;
extern crate num_traits;
extern crate rand;

extern crate nnet;

extern crate anna_eval;
extern crate anna_learning;
extern crate anna_model;
extern crate anna_simulation;
extern crate anna_utils;

use nnet::Network;
use rand::{SeedableRng, StdRng};

use std::{
  collections::{HashMap, HashSet},
  path::Path,
};

use anna_eval::Eval;

use anna_learning::{
  agent_episode::{players_clean, players_create, Episode, Policy},
  qnet::QNet,
  snet::SNet,
};

use anna_model::{
  cards::{Card, CARDS},
  encoders::CardsEncoder,
  money::*,
  ActionClass, SeatId,
};

use anna_simulation::{
  engine, players::Players, players_sel::PlayersSel, rounds::seat_next, Event, Sim, State,
};

use anna_utils as utils;
use anna_utils::{bincode, logging, random};

use console_player::{PlayerUI, StateUI};
use ui;

const SEATS: usize = 6;

pub fn main(path_data: &Path, task: Option<String>) {
  let ref path_synthetic = path_data.join("synthetic");
  let ref path_nets = path_data.join("nets");

  match task {
    Some(task) => {
      if task == String::from("keys_dump") {
        let _ = ncurses::initscr();
        ncurses::noecho();
        let mut ch = ncurses::getch();
        while ch != 4 {
          println!("{:?}", ch);
          ch = ncurses::getch();
        }
        ui::close();
      } else if task == String::from("cards_select") {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        let ref hotkeys = ui::hk_load();
        let ref windows = ui::open();

        let n = 6;

        let mut running = true;

        let mut cards = CARDS;

        let mut total = 0;
        let mut errors = 0;

        rng.shuffle(&mut cards);

        while running {
          let ref cards_expected: Vec<Card> = cards.iter().take(n).map(|&c| c).collect();

          let mut expected = String::from("");
          for c in cards_expected {
            expected.extend(c.to_string().chars());
            expected.push(' ');
          }

          ncurses::wclear(windows.console);
          ncurses::wprintw(windows.console, &format!("cards:\t{}\n", expected));

          ncurses::wprintw(windows.console, "cards:\t");
          ncurses::wrefresh(windows.console);

          match ui::select_cards(hotkeys, windows, n) {
            Ok(xs) => {
              total += 1;
              if xs != *cards_expected {
                errors += 1
              } else {
                rng.shuffle(&mut cards);
              };
              ncurses::wprintw(windows.console, &format!("\n{} / {}\n\n", errors, total));
            }
            _ => {
              running = false;
            }
          }
        }

        ui::close();
      } else {
        println!("Invalid task: {}", task);
      }
    }
    None => {
      let ref mut thread_rng = rand::thread_rng();
      let ref path_graphs = path_data.join("graphs");
      let ref path_networks = path_data.join("networks");

      use anna_learning::plan::*;
      let plan = plan_texas_limit_n(6, &path_synthetic);

      let ref sim = plan.sim(true);

      let snet = SNet::new(path_graphs, &plan.profile, &plan.eval, &plan.snet_hiddens).unwrap();

      let cards_encoder_create =
        |rng| match plan.cards_encoders(rng, &snet, &plan.eval, path_networks) {
          CardsEncoders::StrengthNNet(encoder) => encoder,
          _ => unimplemented!(),
        };

      let actions_encoder = plan.actions_encoder();

      let cards_encoder = cards_encoder_create(random::rseed(thread_rng));

      let qnet =
        QNet::new(path_graphs, sim, actions_encoder.as_ref(), &cards_encoder, &plan.qnet_hiddens)
          .unwrap();

      let ref hotkeys = ui::hk_load();
      let ref windows = ui::open();

      let mut running = true;
      while running {
        let result = {
          let mut players =
            players_create(path_networks, &qnet, actions_encoder.as_ref(), cards_encoder_create);
          run(sim, &plan.eval, hotkeys, windows, &mut players)
        };

        match result {
          Err(ui::Exit::Quit) => {
            running = false;
          }
          _ => {
            // NOOP
            ui::log(windows.console, "\n--------------------------------");
            ui::log(windows.console, "");
          }
        }
      }

      ui::close();
    }
  }
}

fn game_run<
  'a,
  CE: CardsEncoder,
  GS: FnMut(
    &mut StateUI<PlayersSel<Episode<'a, StdRng, CE>>>,
    Vec<(usize, Money)>,
  ) -> Result<(), ui::Exit>,
  RS: FnMut(
    &mut StateUI<PlayersSel<Episode<'a, StdRng, CE>>>,
    usize,
    Money,
  ) -> Result<Vec<Card>, ui::Exit>,
  IN: FnMut(
    &mut StateUI<PlayersSel<Episode<'a, StdRng, CE>>>,
    &[Money],
    usize,
    &[Money],
    &Vec<(usize, Vec<Card>)>,
  ) -> (),
  PL: FnMut(
    &mut StateUI<PlayersSel<Episode<'a, StdRng, CE>>>,
    usize,
    Money,
    Option<Money>,
    &[Money],
    usize,
    Money,
    &[Event],
  ) -> Result<u8, ui::Exit>,
>(
  sim: &Sim,
  evaluator: &Eval,
  hotkeys: &ui::HotKeys,
  windows: &ui::Windows,
  dealer: SeatId,
  player_ui: &mut PlayerUI<'a, PlayersSel<Episode<'a, StdRng, CE>>, GS, RS, IN, PL>,
) -> Result<(), ui::Exit> {
  use num_traits::FromPrimitive;

  // Select hand.
  ncurses::wclear(windows.table);
  let (hand_l, hand_r) = ui::select_hand(hotkeys, windows)?;
  assert!(hand_l != hand_r);

  // Compute who is playing
  let mut playing = HashSet::new();
  for player_id in 0..sim.profile.players {
    if (player_ui.state.player_funds[player_id] - sim.blind_biggest) >= 0 {
      playing.insert(player_id);
    }
  }

  // Compute small blind seat location.
  let mut blind_small_seat_id: SeatId = dealer;
  blind_small_seat_id = {
    let mut x = seat_next(SEATS, blind_small_seat_id);
    while !playing.contains(&x) {
      x = seat_next(SEATS, x);
    }
    x
  };

  // Clean and setup PlayersSel
  players_clean(player_ui.state.players);
  player_ui.state.players.setup(&playing);
  blind_small_seat_id = player_ui.state.players.seat_of(blind_small_seat_id).unwrap();

  // Create initial hands, for simulation purpose.
  let mut players_init: Vec<(SeatId, Money, Vec<Card>)> = Vec::new();
  for &player_id in playing.iter() {
    let seat_id = player_ui.state.players.seat_of(player_id).unwrap();
    let hand = if player_id == 0 { vec![hand_l, hand_r] } else { vec![CARDS[0], CARDS[0]] };
    let funds = player_ui.state.player_funds[player_id];
    players_init.push((seat_id, funds, hand));
  }

  // Playing game.
  let (rounds, events, result) =
    engine::table_game_simulate_(sim, evaluator, player_ui, &players_init, blind_small_seat_id)
      .map_err(|(_, _, e)| match e {
        engine::Error::Play(err) => err,
        err => panic!("Fatal error during replay: {:?}", err),
      })?;

  let score = match result {
    None => {
      ui::log(windows.console, "abort");
    }
    Some((player_funds, player_pots, mut player_states)) => {
      for (seat_id, &player_id) in player_ui.state.players.locations.iter().enumerate() {
        player_ui.state.player_funds[player_id] = player_funds[seat_id];
        player_ui.state.player_pots[player_id] = player_pots[seat_id];
      }

      let finalists: usize = player_states
        .iter()
        .enumerate()
        .filter(|&(i, s)| match s {
          &State::Play { .. } => true,
          _ => false,
        })
        .count();

      if finalists > 1 {
        // Select remaining table cards.
        if player_ui.state.table_cards.len() < 5 {
          let n = 5 - player_ui.state.table_cards.len();

          ncurses::wprintw(windows.console, &format!("table.cards[{}]:\t", n));
          ncurses::wrefresh(windows.console);
          let xs = ui::select_cards(hotkeys, windows, n)?;
          ncurses::wprintw(windows.console, "\n");

          player_ui.state.table_cards.extend(xs.iter());
        }

        // Select finalists hands
        for (seat_id, &player_id) in player_ui.state.players.locations.iter().skip(1).enumerate() {
          let state = match player_states[seat_id] {
            State::Play { ref cards, all_in } => {
              ui::table_render(
                windows,
                &player_ui.state.player_funds,
                &player_ui.state.player_pots,
                &playing,
                Some(player_id),
              );
              ncurses::wprintw(windows.console, &format!("player.{}.hand:\t", player_id + 1));
              ncurses::wrefresh(windows.console);
              let xs = ui::select_cards(hotkeys, windows, 2)?;
              ncurses::wprintw(windows.console, "\n");
              Some(State::Play { cards: vec![xs[0], xs[1]], all_in })
            }
            _ => None,
          };
          for s in state {
            player_states[seat_id] = s;
          }
        }
      }

      ui::log(windows.console, "");

      let (_, ref winners_seats, _) =
        engine::game_finish(evaluator, &player_ui.state.table_cards, &player_states);

      let winners: Vec<usize> = winners_seats
        .iter()
        .map(|&seat_id| player_ui.state.players.player_at(seat_id).unwrap())
        .collect();

      ui::log(windows.console, format!("winners: {:?}", winners));

      // Update funds.
      engine::funds_update(
        &sim,
        &mut player_ui.state.player_funds,
        &player_ui.state.player_pots,
        &winners,
      );
    }
  };

  Ok(())
}

fn run<'a, CE: CardsEncoder>(
  sim: &Sim,
  evaluator: &Eval,
  hotkeys: &ui::HotKeys,
  windows: &ui::Windows,
  players_init: &'a mut PlayersSel<Episode<'a, StdRng, CE>>,
) -> Result<(), ui::Exit> {
  let (blind_small, blind_big) = ui::select_blinds(hotkeys, windows)?;

  let player_funds_init = ui::select_funds(windows, blind_big)?;
  ui::log(windows.console, format!("players: {:?}", player_funds_init.len()));

  if player_funds_init.len() < 2 || player_funds_init.iter().position(|&(s, _)| s == 0).is_none() {
    return Err(ui::Exit::Restart);
  }

  let mut player_ui = PlayerUI {
    state: StateUI {
      players: players_init,
      player_funds: vec![MONEY_ZERO; 6],
      player_pots: vec![MONEY_ZERO; 6],
      table_cards: Vec::new(),
      table_target_init: MONEY_ZERO,
    },

    game_start: move |state, blinds| {
      for (seat_id, blind) in blinds {
        state.player_pots[state.players.player_at(seat_id).unwrap()] = blind;
      }
      Ok(())
    },
    round_start: move |state, round_id, table_target| {
      ui::table_render(
        windows,
        &state.player_funds,
        &state.player_pots,
        &state.players.locations.iter().cloned().collect(),
        None,
      );

      if round_id > 0 {
        state.table_target_init = table_target;
      }

      let cards = match round_id {
        0 => Vec::new(),
        _ => {
          ncurses::wprintw(windows.console, &format!("round.{}:  \t", round_id));
          ncurses::wrefresh(windows.console);

          let len = match round_id {
            1 => 3,
            2 => 1,
            3 => 1,
            _ => {
              panic!("invalid round_id: {}", round_id);
            }
          };

          let xs = ui::select_cards(hotkeys, windows, len)?;
          ncurses::wprintw(windows.console, "\n");
          ncurses::wrefresh(windows.console);

          xs
        }
      };

      state.table_cards.extend(cards.iter());

      Ok(state.table_cards.clone())
    },
    init: move |state, blind_big, seat_sb, player_funds, playing_hands| {
      let xs = playing_hands
        .iter()
        .filter(|&&(player_id, _)| player_id == 0)
        .map(|(x, cs)| (state.players.seat_of(*x).unwrap(), cs.clone()))
        .collect();
      state.players.init(blind_big, seat_sb, &player_funds, &xs);
    },
    play: move |state,
                round_id,
                table_target,
                table_target_raise,
                player_pots,
                seat_id,
                player_fund,
                events| {
      let player_id = state.players.player_at(seat_id).unwrap();
      let player_pot = player_pots[player_id];

      ui::table_render(
        windows,
        &state.player_funds,
        &state.player_pots,
        &state.players.locations.iter().cloned().collect(),
        Some(player_id),
      );

      let p = if seat_id == 0 {
        let class = state
          .players
          .play(
            round_id,
            table_target,
            table_target_raise,
            player_pots,
            seat_id,
            player_fund,
            events,
          )
          .expect("Internal Players failure.");

        Some((sim.action_class.to_action(class), class as usize))
      } else {
        None
      };

      let action_i = ui::select_action(
        sim.action_class,
        windows,
        blind_big,
        round_id,
        state.table_target_init,
        table_target,
        table_target_raise,
        player_pot,
        player_fund,
        p,
      )?;

      let (action, pledge) = sim
        .action_class
        .unapply(
          blind_big,
          round_id,
          player_fund,
          table_target,
          table_target_raise.unwrap_or(Money::zero()),
          player_pot,
          action_i as u8,
        )
        .expect("Invalid action class.");

      for money in pledge {
        state.player_funds[player_id] = Money::from_u32((player_fund - money) as u32);
        state.player_pots[player_id] = player_pot + money;
      }

      Ok(action_i as u8)
    },
  };

  // Apply initial funds.
  let mut players_init: Vec<(SeatId, Money, (Card, Card))> = Vec::new();
  for &(player_id, funds) in player_funds_init.iter() {
    player_ui.state.player_funds[player_id] = funds;
  }

  let running = true;
  let mut dealer_p: Option<SeatId> = None;

  while running {
    // Select dealer (or update a seat fund).
    let dealer = {
      ui::table_render(
        windows,
        &player_ui.state.player_funds,
        &player_ui.state.player_pots,
        &HashSet::new(),
        None,
      );
      let (mut seat_id, mut shift) = ui::select_dealer(hotkeys, windows, dealer_p)?;
      while shift {
        let m = ui::select_fund(windows, blind_big)?;
        player_ui.state.player_funds[seat_id as usize] = m;
        let (seat_id0, shift0) = ui::select_dealer(hotkeys, windows, dealer_p)?;
        seat_id = seat_id0;
        shift = shift0;
      }
      seat_id
    };

    ui::log(windows.console, "");

    // Run game
    match game_run(sim, evaluator, hotkeys, windows, dealer, &mut player_ui) {
      Ok(_) => {}
      Err(ui::Exit::Restart) => {}
      Err(err) => {
        return Err(err);
      }
    }

    // Reset state
    player_ui.state.player_pots = vec![MONEY_ZERO; 6];
    player_ui.state.table_cards = Vec::new();
    player_ui.state.table_target_init = MONEY_ZERO;

    ui::log(windows.console, "");
    ui::log(windows.console, "--------------------------------");

    dealer_p = Some(seat_next(SEATS, dealer));
    while player_ui.state.player_funds[dealer_p.unwrap() as usize] == MONEY_ZERO
      && dealer_p != Some(dealer)
    {
      dealer_p = Some(seat_next(SEATS, dealer_p.unwrap()));
    }
  }

  Ok(())
}
