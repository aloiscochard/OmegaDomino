extern crate serde;

use self::serde::{
  de::{Error, Visitor},
  Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

pub fn cards_class(deck: &Vec<Card>, cards: &Vec<Card>) -> Vec<usize> {
  cards
    .iter()
    .map(|c| deck.binary_search(c).expect(&format!("Card not part of deck. ({:?})", c)))
    .collect()
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Card {
  pub value: CardVal,
  pub suit: Suit,
}

impl Card {
  pub fn from_u8(u: u8) -> Option<Card> {
    use std::ops::Div;
    if u < 52 {
      let suit_id = u.div(13);
      let suit = SUITS[suit_id as usize];
      let card_val_id = u - (suit_id * 13);
      let card_val = CARD_VALS[card_val_id as usize];
      Some(Card { suit: suit, value: card_val })
    } else {
      None
    }
  }

  pub fn as_u8(self) -> u8 {
    self.value as u8 + (self.suit as u8 * 13)
  }
}

impl fmt::Display for Card {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{}{}", self.value, self.suit)
  }
}

impl<'de> Deserialize<'de> for Card {
  fn deserialize<D>(d: D) -> Result<Card, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct CardVisitor;

    impl<'de> Visitor<'de> for CardVisitor {
      type Value = Card;

      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer between 0 and 51")
      }

      fn visit_u8<E>(self, value: u8) -> Result<Card, E>
      where
        E: Error,
      {
        match Card::from_u8(value) {
          Some(card) => Ok(card),
          None => Err(E::custom(format!("u8 out of range for card: {}", value))),
        }
      }
    }

    d.deserialize_u8(CardVisitor)
  }
}

impl Serialize for Card {
  fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    s.serialize_u8(Card::as_u8(*self))
  }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CardVal {
  C2,
  C3,
  C4,
  C5,
  C6,
  C7,
  C8,
  C9,
  C10,
  CJ,
  CQ,
  CK,
  CA,
}
impl fmt::Display for CardVal {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match *self {
      CardVal::CJ => write!(f, "J"),
      CardVal::CQ => write!(f, "Q"),
      CardVal::CK => write!(f, "K"),
      CardVal::CA => write!(f, "A"),
      value => f.write_str((value as usize + 2).to_string().as_str()),
    }
  }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Suit {
  Spade,
  Heart,
  Diamond,
  Club,
}

impl fmt::Display for Suit {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    use self::Suit::*;
    f.write_str(match self {
      &Spade => "♠",
      &Heart => "♥",
      &Diamond => "♦",
      &Club => "♣",
    })
  }
}

pub const CARDS: [Card; 52] = [
  Card { suit: Suit::Spade, value: CardVal::C2 },
  Card { suit: Suit::Spade, value: CardVal::C3 },
  Card { suit: Suit::Spade, value: CardVal::C4 },
  Card { suit: Suit::Spade, value: CardVal::C5 },
  Card { suit: Suit::Spade, value: CardVal::C6 },
  Card { suit: Suit::Spade, value: CardVal::C7 },
  Card { suit: Suit::Spade, value: CardVal::C8 },
  Card { suit: Suit::Spade, value: CardVal::C9 },
  Card { suit: Suit::Spade, value: CardVal::C10 },
  Card { suit: Suit::Spade, value: CardVal::CJ },
  Card { suit: Suit::Spade, value: CardVal::CQ },
  Card { suit: Suit::Spade, value: CardVal::CK },
  Card { suit: Suit::Spade, value: CardVal::CA },
  Card { suit: Suit::Heart, value: CardVal::C2 },
  Card { suit: Suit::Heart, value: CardVal::C3 },
  Card { suit: Suit::Heart, value: CardVal::C4 },
  Card { suit: Suit::Heart, value: CardVal::C5 },
  Card { suit: Suit::Heart, value: CardVal::C6 },
  Card { suit: Suit::Heart, value: CardVal::C7 },
  Card { suit: Suit::Heart, value: CardVal::C8 },
  Card { suit: Suit::Heart, value: CardVal::C9 },
  Card { suit: Suit::Heart, value: CardVal::C10 },
  Card { suit: Suit::Heart, value: CardVal::CJ },
  Card { suit: Suit::Heart, value: CardVal::CQ },
  Card { suit: Suit::Heart, value: CardVal::CK },
  Card { suit: Suit::Heart, value: CardVal::CA },
  Card { suit: Suit::Diamond, value: CardVal::C2 },
  Card { suit: Suit::Diamond, value: CardVal::C3 },
  Card { suit: Suit::Diamond, value: CardVal::C4 },
  Card { suit: Suit::Diamond, value: CardVal::C5 },
  Card { suit: Suit::Diamond, value: CardVal::C6 },
  Card { suit: Suit::Diamond, value: CardVal::C7 },
  Card { suit: Suit::Diamond, value: CardVal::C8 },
  Card { suit: Suit::Diamond, value: CardVal::C9 },
  Card { suit: Suit::Diamond, value: CardVal::C10 },
  Card { suit: Suit::Diamond, value: CardVal::CJ },
  Card { suit: Suit::Diamond, value: CardVal::CQ },
  Card { suit: Suit::Diamond, value: CardVal::CK },
  Card { suit: Suit::Diamond, value: CardVal::CA },
  Card { suit: Suit::Club, value: CardVal::C2 },
  Card { suit: Suit::Club, value: CardVal::C3 },
  Card { suit: Suit::Club, value: CardVal::C4 },
  Card { suit: Suit::Club, value: CardVal::C5 },
  Card { suit: Suit::Club, value: CardVal::C6 },
  Card { suit: Suit::Club, value: CardVal::C7 },
  Card { suit: Suit::Club, value: CardVal::C8 },
  Card { suit: Suit::Club, value: CardVal::C9 },
  Card { suit: Suit::Club, value: CardVal::C10 },
  Card { suit: Suit::Club, value: CardVal::CJ },
  Card { suit: Suit::Club, value: CardVal::CQ },
  Card { suit: Suit::Club, value: CardVal::CK },
  Card { suit: Suit::Club, value: CardVal::CA },
];

pub const CARD_VALS: [CardVal; 13] = [
  CardVal::C2,
  CardVal::C3,
  CardVal::C4,
  CardVal::C5,
  CardVal::C6,
  CardVal::C7,
  CardVal::C8,
  CardVal::C9,
  CardVal::C10,
  CardVal::CJ,
  CardVal::CQ,
  CardVal::CK,
  CardVal::CA,
];

pub const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

pub const KUHN_CARDS: [Card; 3] = [
  Card { suit: Suit::Spade, value: CardVal::CJ },
  Card { suit: Suit::Spade, value: CardVal::CQ },
  Card { suit: Suit::Spade, value: CardVal::CK },
];

pub const LEDUC_CARDS: [Card; 6] = [
  Card { suit: Suit::Spade, value: CardVal::CJ },
  Card { suit: Suit::Heart, value: CardVal::CJ },
  Card { suit: Suit::Spade, value: CardVal::CQ },
  Card { suit: Suit::Heart, value: CardVal::CQ },
  Card { suit: Suit::Spade, value: CardVal::CK },
  Card { suit: Suit::Heart, value: CardVal::CK },
];
