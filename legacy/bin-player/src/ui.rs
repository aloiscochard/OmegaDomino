use std::collections::{HashMap, HashSet};

use std::{fmt, thread, time};

use ncurses::*;

use anna_model::{
  cards::{Card, CardVal, Suit, CARD_VALS, SUITS},
  money::*,
  Action, ActionClass, SeatId,
};
use anna_simulation::rounds;

#[derive(Clone, Debug)]
pub enum Exit {
  Quit,
  Restart,
}

pub struct HotKeys {
  pub big_blinds: HashMap<usize, (String, Money)>,
  pub card_vals: HashMap<usize, (String, CardVal)>,
  pub seats: HashMap<usize, (String, SeatId)>,
  pub suits: HashMap<usize, (String, Suit)>,
}

pub struct Windows {
  pub console: WINDOW,
  pub hotkeys: WINDOW,
  pub table: WINDOW,
}

// HotKeys
const HOTKEYS: [i32; 16] =
  [033, 091, 123, 040, 039, 044, 046, 112, 097, 111, 101, 117, 059, 113, 106, 107];

const HOTKEYS_SHIFT: [i32; 16] = [49, 50, 51, 52, 34, 60, 62, 80, 65, 79, 69, 85, 58, 81, 74, 75];

pub const ACTIONS_HK: [usize; 3] = [15, 11, 07];

const BIG_BLINDS: [u32; 16] =
  [2, 4, 5, 10, 20, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 5000];
const BIG_BLINDS_HK: [usize; 16] = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3];

const CARD_VALS_HK: [usize; 13] = [12, 13, 14, 15, 8, 9, 10, 11, 7, 0, 1, 2, 3];

const FUNDS_RATIOS: [u32; 16] = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987];
const FUNDS_RATIOS_HK: [usize; 16] = [15, 12, 13, 14, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3];

pub const PLEGES_HK: [usize; 16] = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3];

const SEATS: usize = 6;
const SEATS_HK: [usize; 6] = [09, 10, 11, 05, 06, 07];

// Style and layout
const CP_DEFAULT: i16 = 0;
const CP_ENABLED: i16 = 1;
const CP_DISABLED: i16 = 8;

const SCREEN_WIDTH: usize = 52;
const SCREEN_HEIGHT_MIN: usize = 32;

pub fn hk_actions(
  action_class: &ActionClass,
  blind_big: Money,
  round_id: usize,
  table_target: Money,
  table_target_raise: Option<Money>,
  player_pot: Money,
  player_fund: Money,
) -> (Vec<f32>, HashMap<usize, (String, usize)>) {
  let mut xs = HashMap::new();
  let mut action_probs: Vec<f32> = vec![1.; 18];

  let mask = action_class.normalize(
    blind_big,
    round_id,
    table_target,
    table_target_raise,
    player_fund,
    player_pot,
  );

  let callable = mask.contains(&1);
  let raisable = mask.contains(&2);

  if action_probs[0] != 0. {
    xs.insert(ACTIONS_HK[0], (String::from("Fold"), 0));
  }

  if action_probs[1] != 0. && callable {
    if table_target == player_pot {
      xs.insert(ACTIONS_HK[1], (String::from("Check"), 1));
    } else {
      xs.insert(ACTIONS_HK[1], (String::from("Call"), 1));
    }
  }

  if raisable {
    xs.insert(ACTIONS_HK[2], (String::from("Raise"), 2));
  }

  (action_probs, xs)
}

/*
pub fn hk_action_pledges(blind_big: Money, player_fund: Money, player_pot: Money,
                         table_target: Money, table_target_raise: Money,
                         action_probs: &Vec<f32>, table_target_init: Option<Money>)
                         -> HashMap<usize, (String, usize)> {
  let mut xs = HashMap::new();

  let delta = Money::from_u32((table_target - player_pot) as u32);
  let raise_class_min =
    pledge_ratio_classify(blind_big, money_add(delta, table_target_raise)).unwrap_or(14);

  for i in 0..16 {
    if action_probs[i + 2] != 0. && i >= raise_class_min {
      let lbl = match i {
        15 => String::from("AllIn"),
        i => {
          let m0 = pledge_unclassify(blind_big, player_fund, i as u8);

          let m1 = match table_target_init {
            None => m0,
            Some(table_target_init) => {
              let delta = money_diff(player_pot, table_target_init);
              if delta > 0 {
                money_add(m0, money_from_raw(delta as u32))
              } else {
                m0
              }
            }
          };

          m1.to_string()
        }
      };
      xs.insert(PLEGES_HK[i], (lbl, i + 2));
    }
  }

  xs
}
*/

pub fn hk_funds(blind_big: Money) -> HashMap<usize, (String, Money)> {
  let mut xs = HashMap::new();

  for (i, ratio) in FUNDS_RATIOS.iter().enumerate() {
    let hk = FUNDS_RATIOS_HK[i];
    let m = blind_big * (*ratio);
    let lbl = if m == MONEY_ZERO { String::from("OFF") } else { m.to_string() };
    xs.insert(hk, (lbl, m));
  }

  xs
}

pub fn hk_load() -> HotKeys {
  HotKeys {
    card_vals: {
      use self::CardVal::*;
      let mut xs = HashMap::new();

      for &card_val in CARD_VALS.iter() {
        xs.insert(CARD_VALS_HK[card_val as usize], (card_val.to_string(), card_val));
      }

      xs
    },
    big_blinds: {
      let mut xs = HashMap::new();

      for (i, &bb) in BIG_BLINDS.iter().enumerate() {
        let m = Money::from_u32(bb);
        xs.insert(BIG_BLINDS_HK[i], (m.to_string(), m));
      }

      xs
    },
    seats: {
      let mut xs = HashMap::new();

      xs.insert(SEATS_HK[0], (1.to_string(), 0));
      xs.insert(SEATS_HK[1], (2.to_string(), 1));
      xs.insert(SEATS_HK[2], (3.to_string(), 2));
      xs.insert(SEATS_HK[3], (4.to_string(), 3));
      xs.insert(SEATS_HK[4], (5.to_string(), 4));
      xs.insert(SEATS_HK[5], (6.to_string(), 5));

      xs
    },
    suits: {
      let mut xs = HashMap::new();

      for &suit in SUITS.iter() {
        xs.insert(suit as usize + 8, (suit.to_string(), suit));
      }

      xs
    },
  }
}

pub fn open() -> Windows {
  // Initialize UI
  let locale_conf = LcCategory::all;
  setlocale(locale_conf, "en_US.UTF-8"); // if your locale is like mine(zh_CN.UTF-8).

  let _ = initscr();
  start_color();
  raw();

  noecho();
  curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
  refresh();

  // Compute layout according to screen size.
  let mut screen_width = 0;
  let mut screen_height = 0;
  getmaxyx(stdscr(), &mut screen_height, &mut screen_width);

  if screen_width < SCREEN_WIDTH as i32 || screen_height < SCREEN_HEIGHT_MIN as i32 {
    panic!("Terminal is too small. (minimum {}w {}h)", screen_width, screen_height);
  }

  // Color scheme
  init_pair(CP_ENABLED, COLOR_BLUE, COLOR_BLACK);
  init_pair(CP_DISABLED, 12, COLOR_BLACK);

  // Windows
  let win_console = newwin(screen_height - 24, 52, 0, (screen_width / 2) - 25); // h, w, y, x
  scrollok(win_console, true);
  for _ in 0..(screen_height - 24) {
    wprintw(win_console, "\n");
  }
  wrefresh(win_console);

  let win_hotkeys = newwin(12, 45, screen_height - 24, (screen_width / 2) - 22); // h, w, y, x
  let win_table = newwin(12, 52, screen_height - 12, (screen_width / 2) - 25); // h, w, y, x

  Windows { console: win_console, hotkeys: win_hotkeys, table: win_table }
}

pub fn close() {
  endwin();
}

pub fn hotkeys_draw<A>(
  win: WINDOW,
  hotkeys: &HashMap<usize, (String, A)>,
  highlight: Option<usize>,
) -> () {
  wclear(win);

  let attr_color = move |i: usize| {
    let hk = hotkeys.get(&i);
    COLOR_PAIR(if Some(i) == highlight {
      CP_ENABLED
    } else if hk.is_some() {
      CP_DEFAULT
    } else {
      CP_DISABLED
    })
  };

  for l in 0..4 {
    for c in 0..4 {
      let i = (l * 4) + c;
      let attr = attr_color(i);
      wattron(win, attr);
      waddch(win, ACS_ULCORNER());
      for _ in 0..9 {
        waddch(win, ACS_HLINE());
      }
      waddch(win, ACS_URCORNER());
      wattroff(win, attr);
    }

    wprintw(win, "\n");

    for c in 0..4 {
      let i = (l * 4) + c;
      let attr = attr_color(i);
      wattron(win, attr);
      waddch(win, ACS_VLINE());
      let hk = hotkeys.get(&i);
      let label =
        hk.map(|&(ref key, _)| format!("{: ^9}", key)).unwrap_or(String::from("         "));
      wprintw(win, &label);
      waddch(win, ACS_VLINE());
      wattroff(win, attr);
    }

    wprintw(win, "\n");

    for c in 0..4 {
      let i = (l * 4) + c;
      let attr = attr_color(i);
      wattron(win, attr);
      waddch(win, ACS_LLCORNER());
      for _ in 0..9 {
        waddch(win, ACS_HLINE());
      }
      waddch(win, ACS_LRCORNER());
      wattroff(win, attr);
    }

    wprintw(win, "\n");
  }

  wrefresh(win);
}

pub fn hotkey_get<A>(
  win: WINDOW,
  hotkeys: &HashMap<usize, (String, A)>,
  highlight: Option<usize>,
) -> Result<&A, Exit> {
  hotkey_get_(win, hotkeys, highlight).map(|(a, _)| a)
}

pub fn hotkey_get_<A>(
  win: WINDOW,
  hotkeys: &HashMap<usize, (String, A)>,
  highlight: Option<usize>,
) -> Result<(&A, bool), Exit> {
  hotkeys_draw(win, hotkeys, highlight);
  loop {
    let (i, shift) = hotkey_read()?;
    match highlight {
      Some(hl) if i == 128 => {
        return Ok((&hotkeys.get(&hl).expect("Highlight not found in hotkeys.").1, shift));
      }
      _ => {
        for &(_, ref value) in hotkeys.get(&i) {
          hotkeys_draw(win, hotkeys, Some(i));
          thread::sleep(time::Duration::from_millis(50));
          return Ok((value, shift));
        }
      }
    }
  }
}

pub fn log<A: fmt::Display>(w: WINDOW, a: A) -> () {
  wprintw(w, &format!("{}\n", a));
  wrefresh(w);
}

pub fn select_action(
  action_class: &ActionClass,
  windows: &Windows,
  blind_big: Money,
  round_id: usize,
  table_target_init: Money,
  table_target: Money,
  table_target_raise: Option<Money>,
  player_pot: Money,
  player_fund: Money,
  highlight: Option<(Action, usize)>,
) -> Result<usize, Exit> {
  let (action_probs, actions) = hk_actions(
    action_class,
    blind_big,
    round_id,
    table_target,
    table_target_raise,
    player_pot,
    player_fund,
  );

  let actions_hl = highlight.map(|(action, action_i)| {
    if action_i < 2 {
      ACTIONS_HK[action_i]
    } else {
      if action == Action::Call {
        ACTIONS_HK[1]
      } else {
        ACTIONS_HK[2]
      }
    }
  });

  let i0 = hotkey_get(windows.hotkeys, &actions, actions_hl)?;

  if *i0 == 1 {
    if action_probs[1] == 0. {
      // Call
      let pledge = Money::from_u32((table_target - player_pot) as u32);
      Ok(action_class.apply(blind_big, player_fund, Action::Call, pledge) as usize)
    } else {
      // Check
      Ok(1)
    }
  } else if *i0 == 2 {
    // Raise
    Ok(2)
  /*
  let action_pledges = hk_action_pledges(blind_big,
                                         player_fund,
                                         player_pot,
                                         table_target,
                                         table_target_raise,
                                         &action_probs,
                                         Some(table_target_init));
  let action_pledges_hl = highlight.and_then(|(action, action_i)| {
                                               if action_i < 2 {
                                                 None
                                               } else {
                                                 Some(PLEGES_HK[action_i - 2])
                                               }
                                             });
  hotkey_get(windows.hotkeys, &action_pledges, action_pledges_hl).map(|x| *x)
  */
  } else {
    Ok(*i0)
  }
}

pub fn select_blinds(hotkeys: &HotKeys, windows: &Windows) -> Result<(Money, Money), Exit> {
  use std::ops::Div;

  wprintw(windows.console, "blinds: ");
  wrefresh(windows.console);
  let bb = hotkey_get(windows.hotkeys, &hotkeys.big_blinds, None)?;
  let mut sb = bb.div(2);
  if *bb > Money::new(1, 00) {
    sb = (sb / 100) * 100;
  }
  wprintw(windows.console, &format!("{} / {}\n", sb, bb));
  wrefresh(windows.console);

  Ok((sb, bb.clone()))
}

pub fn select_card(hotkeys: &HotKeys, windows: &Windows) -> Result<Card, Exit> {
  let card_val = hotkey_get(windows.hotkeys, &hotkeys.card_vals, None)?;
  wprintw(windows.console, &card_val.to_string());
  wrefresh(windows.console);

  let suit = hotkey_get(windows.hotkeys, &hotkeys.suits, None)?;
  wprintw(windows.console, &suit.to_string());
  wrefresh(windows.console);

  Ok(Card { suit: *suit, value: *card_val })
}

pub fn select_cards(hotkeys: &HotKeys, windows: &Windows, n: usize) -> Result<Vec<Card>, Exit> {
  let mut xs = Vec::new();

  for _ in 0..n {
    let x = select_card(hotkeys, windows)?;
    wprintw(windows.console, " ");
    xs.push(x);
  }

  Ok(xs)
}

pub fn select_dealer(
  hotkeys: &HotKeys,
  windows: &Windows,
  hl: Option<SeatId>,
) -> Result<(SeatId, bool), Exit> {
  wprintw(windows.console, "dealer: ");
  wrefresh(windows.console);
  let (seat_id, shift) =
    hotkey_get_(windows.hotkeys, &hotkeys.seats, hl.map(|seat| SEATS_HK[seat as usize]))?;
  log(windows.console, seat_id.to_string());

  Ok((*seat_id, shift))
}

pub fn select_fund(windows: &Windows, blind_big: Money) -> Result<Money, Exit> {
  let hotkeys = hk_funds(blind_big);
  let m = hotkey_get(windows.hotkeys, &hotkeys, None)?;
  Ok(*m)
}

pub fn select_funds(windows: &Windows, blind_big: Money) -> Result<Vec<(SeatId, Money)>, Exit> {
  let hotkeys = hk_funds(blind_big);
  let mut funds = Vec::new();
  let mut seats = HashMap::new();

  for seat_id in 0..SEATS {
    seats.insert(seat_id, String::from("?"));
    table_draw(windows.table, &seats, String::from(""), Some(seat_id));

    let m = hotkey_get(windows.hotkeys, &hotkeys, None)?;

    if *m == MONEY_ZERO {
      seats.remove(&seat_id);
    } else {
      funds.push((seat_id, *m));
      seats.insert(seat_id, format!("{}", m));
    }
  }

  Ok(funds)
}

pub fn select_hand(hotkeys: &HotKeys, windows: &Windows) -> Result<(Card, Card), Exit> {
  wprintw(windows.console, "player.1.hand:\t");
  wrefresh(windows.console);

  let l = select_card(hotkeys, windows)?;

  wprintw(windows.console, " ");

  let r = select_card(hotkeys, windows)?;

  wprintw(windows.console, "\n");
  wrefresh(windows.console);

  Ok((l, r))
}

pub fn table_render(
  windows: &Windows,
  player_funds: &[Money],
  player_pots: &[Money],
  playing: &HashSet<usize>,
  active: Option<usize>,
) -> () {
  let mut seat_labels = HashMap::new();
  for player_id in 0..SEATS {
    if playing.contains(&player_id) {
      let lbl_l = format!("{: ^9}", player_funds[player_id].to_string());
      let lbl_r = format!("{: ^9}", player_pots[player_id].to_string());
      seat_labels.insert(player_id, format!("{}|{}", lbl_l, lbl_r));
    }
  }

  let pot: u32 = player_pots.iter().map(|m| (*m).unpack()).sum();

  table_draw(windows.table, &seat_labels, Money::from_u32(pot).to_string(), active);
}

fn table_draw(
  win: WINDOW,
  seats: &HashMap<usize, String>,
  table: String,
  highlight: Option<usize>,
) -> () {
  wclear(win);

  let boxes_draw = move |pad: usize, xs: &[(String, attr_t)], title: String| {
    for _ in 0..pad {
      wprintw(win, " ");
    }

    let vline = move |color| {
      wattron(win, color);
      waddch(win, ACS_VLINE());
      wattroff(win, color);
    };

    for (i, &(_, color)) in xs.iter().enumerate() {
      if i > 0 {
        for _ in 0..9 {
          wprintw(win, " ");
        }
      }
      wattron(win, color);
      waddch(win, ACS_ULCORNER());
      for _ in 0..19 {
        waddch(win, ACS_HLINE());
      }
      waddch(win, ACS_URCORNER());
      wattroff(win, color);
    }

    wprintw(win, "\n");
    for _ in 0..pad {
      wprintw(win, " ");
    }

    for (i, &(ref lbl, color)) in xs.iter().enumerate() {
      if i > 0 {
        wprintw(win, &title);
      }
      vline(color);
      wprintw(win, &format!("{: ^19}", lbl));
      vline(color);
    }

    wprintw(win, "\n");
    for _ in 0..pad {
      wprintw(win, " ");
    }

    for (i, &(_, color)) in xs.iter().enumerate() {
      if i > 0 {
        for _ in 0..9 {
          wprintw(win, " ");
        }
      }
      wattron(win, color);
      waddch(win, ACS_LLCORNER());
      for _ in 0..19 {
        waddch(win, ACS_HLINE());
      }
      waddch(win, ACS_LRCORNER());
      wattroff(win, color);
    }

    wprintw(win, "\n");
  };

  let get = move |seat_id| {
    let (label, mut attr) = match seats.get(seat_id) {
      None => ("", COLOR_PAIR(CP_DISABLED)),
      Some(ref lbl) => (lbl.as_str(), COLOR_PAIR(CP_DEFAULT)),
    };
    if Some(*seat_id) == highlight {
      attr = COLOR_PAIR(CP_ENABLED);
    }
    (String::from(label), attr)
  };

  boxes_draw(15, &[get(&3)], String::from(""));
  boxes_draw(0, &[get(&2), get(&4)], format!("{: ^9}", table));
  boxes_draw(0, &[get(&1), get(&5)], String::from("         "));
  boxes_draw(15, &[get(&0)], String::from(""));

  wrefresh(win);
}

fn hotkey_read() -> Result<(usize, bool), Exit> {
  loop {
    let ch = getch();
    if ch == 17 {
      // Ctrl+Q = Exit client
      return Err(Exit::Quit);
    } else if ch == 127 {
      // Backspace = Restart
      return Err(Exit::Restart);
    } else if ch == 10 {
      return Ok((128, false));
    } else {
      for i in HOTKEYS.iter().position(|&k| k == ch) {
        return Ok((i, false));
      }
      for i in HOTKEYS_SHIFT.iter().position(|&k| k == ch) {
        return Ok((i, true));
      }
    }
  }
}
