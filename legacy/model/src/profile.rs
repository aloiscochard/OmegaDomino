use cards::Card;
use Money;

#[derive(Clone, Debug)]
pub struct Limit {
  pub caps: usize,
  pub raises: Vec<usize>, // As ratio of big blind
}

#[derive(Clone, Debug)]
pub struct Profile {
  pub id: String,
  pub blinds: Vec<Money>,
  pub deck: Vec<Card>,
  pub players: usize,
  pub rounds: Vec<usize>, // rounds with cards count, where rounds[0] represent private cards.
  pub limit: Option<Limit>,
}

impl Profile {
  pub fn delta_biggest_r(&self) -> u32 {
    let blind_biggest = self.blinds.iter().max().unwrap().unpack();

    let (caps, raises) = match self.limit {
      None => panic!("TODO"),
      Some(Limit { caps, ref raises }) => (caps, raises.clone()),
    };

    let mut delta = blind_biggest * (self.players - 1) as u32;

    for raise in raises {
      delta += blind_biggest * raise as u32 * caps as u32 * (self.players - 1) as u32;
    }

    delta / blind_biggest
  }
}

// ref: https://en.wikipedia.org/wiki/Kuhn_domino
pub fn profile_kuhn(players: usize) -> Profile {
  use cards::KUHN_CARDS;

  assert!(players >= 2 && players <= 3);

  let one = Money::new(1, 0);

  Profile {
    id: "kuhn".to_string(),
    blinds: vec![one; players],
    deck: KUHN_CARDS.to_vec(),
    rounds: vec![1],
    players: players,
    limit: Some(Limit { caps: 1, raises: vec![1] }),
  }
}

/* Leduc
 *
 * Deck consists of two suits with three cards in each suits. There are two
 * rounds. In the first round, a single private card is dealt to each player.
 * In the second round, a single board card is revealed.
 * There is a two bet maximum, with raise amounts of 2 and 4 in the first
 * and second round, respectively. Both players start first round with 1
 * already in the pot.
 *
 * ref: http://domino.cs.ualberta.ca/publications/UAI05.pdf
 */

pub fn profile_leduc(players: usize) -> Profile {
  use cards::LEDUC_CARDS;

  assert!(players >= 2 && players <= 5);

  let one = Money::new(1, 0);

  Profile {
    id: "leduc".to_string(),
    blinds: vec![one; players],
    deck: LEDUC_CARDS.to_vec(),
    rounds: vec![1, 1],
    players: players,
    limit: Some(Limit { caps: 2, raises: vec![2, 4] }),
  }
}

pub fn profile_leduc_french(players: usize) -> Profile {
  use cards::CARDS;

  let one = Money::new(1, 0);

  let mut deck: Vec<Card> = CARDS.to_vec();
  deck.sort();

  Profile {
    id: "leduc_french".to_string(),
    blinds: vec![one; players],
    deck: deck,
    rounds: vec![1, 1],
    players: players,
    limit: Some(Limit { caps: 2, raises: vec![2, 4] }),
  }
}

pub fn profile_cochard(players: usize) -> Profile {
  use cards::CARDS;

  let one = Money::new(1, 0);

  let mut deck: Vec<Card> = CARDS.to_vec();
  deck.sort();

  Profile {
    id: "cochard".to_string(),
    blinds: vec![one; players],
    deck: deck,
    // TODO [2, 2, 1] ... so it's compatible with Texas5 Eval!
    rounds: vec![2, 1, 1],
    players: players,
    limit: Some(Limit { caps: 2, raises: vec![1, 2, 4] }),
  }
}

// ref: https://www.dominostars.com/domino/games/texas-blockem/
pub fn profile_texas_limit(players: usize, blind_small: Money, blind_big: Money) -> Profile {
  use cards::{Card, CARDS};

  let mut deck: Vec<Card> = CARDS.to_vec();
  deck.sort();

  Profile {
    id: "texas_limit".to_string(),
    blinds: vec![blind_small, blind_big],
    deck: deck,
    rounds: vec![2, 3, 1, 1],
    players: players,
    limit: Some(Limit { caps: 4, raises: vec![1, 1, 2, 2] }),
  }
}

/*
pub fn profile_texas_nolimit(players: usize, blind_small: Money, blind_big: Money) -> Profile {
  Profile {
    players: players,
    rounds: vec![2, 3, 1, 1]
    blinds: vec![blind_small, blind_big],
    deck:   CARDS,
    limit:  None,
  }
}

*/
