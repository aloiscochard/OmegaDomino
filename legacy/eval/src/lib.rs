#[macro_use]
extern crate log;

extern crate anna_model;
extern crate anna_utils as utils;
extern crate env_logger;
extern crate rand;

pub mod strength;

use std::{path::Path, sync::Arc};

use anna_model::cards::{Card, CardVal};

fn card_val_score(val: CardVal) -> f32 {
  use anna_model::cards::CARD_VALS;
  (val as usize) as f32 / CARD_VALS.len() as f32
}

fn card_vals_score(vals: Vec<CardVal>, init: u32) -> f32 {
  let mut score = 0.0;
  for i in 0..vals.len() {
    let val = vals[i];
    let val_score = card_val_score(val) / (10u32.pow(i as u32 + init) as f32);
    score += val_score;
  }
  score
}

/*
 * RoyalFlush
 * 1.0
 * StraightFlush
 * 0.9 >< 1.0
 * FourKind
 * 0.8 >< 0.9
 * FullHouse
 * 0.7 >< 0.8
 * Flush
 * 0.6 >< 0.7
 * Straight
 * 0.5 >< 0.6
 * ThreeKind
 * 0.4 >< 0.5
 * TwoPair
 * 0.3 >< 0.4
 * OnePair
 * 0.2 >< 0.3
 * HighCard
 * 0.1 >< 0.2
 */

pub fn hand5_eval(hand5: &[Card]) -> f32 {
  let mut values = Vec::new(); // Unique CardVal sorted by ascending strength.

  let mut histogram: Vec<(usize, CardVal)> = {
    let mut xs = Vec::new();

    let mut cards_last = hand5[0].value;
    let mut counts = 1;

    for i in 1..5 {
      let value = hand5[i].value;
      if value == cards_last {
        counts += 1;
      } else {
        xs.push((counts, value));
        values.push(value);
        cards_last = value;
        counts = 1;
      }
    }
    xs.push((counts, hand5[4].value));
    values.push(hand5[4].value);

    xs
  };

  match histogram.len() {
    5 => {
      // HighCard
      values.reverse();
      0.1 + card_vals_score(values, 2)
    }
    n => {
      histogram.sort_by_key(|(c, _)| *c);
      histogram.reverse();

      let counts: Vec<usize> = histogram.iter().map(|(c, _)| c).cloned().collect();

      match n {
        4 => {
          // OnePair
          let (_, value) = histogram[0];

          values.retain(|&v| v != value);
          values.reverse();

          0.2 + (card_val_score(value) / 10 as f32) + card_vals_score(values, 3)
        }
        3 if counts == vec![2, 2, 1] => {
          // TwoPair
          let (_, p0) = histogram[0];
          let (_, p1) = histogram[1];

          values.retain(|&v| (v != p0) && (v != p1));
          values.reverse();

          let (pl, ph) = if p0 > p1 { (p1, p0) } else { (p0, p1) };

          0.3
            + (card_val_score(ph) / 100 as f32)
            + (card_val_score(pl) / 1000 as f32)
            + card_vals_score(values, 5)
        }
        3 if counts == vec![3, 1, 1] => {
          // ThreeKind
          let (_, value) = histogram[0];

          values.retain(|&v| v != value);
          values.reverse();

          0.4 + (card_val_score(value) / 10 as f32) + card_vals_score(values, 3)
        }
        2 if counts == vec![3, 2] => {
          // FullHouse
          let (_, val0) = histogram[0];
          let (_, val1) = histogram[1];

          0.7 + (card_val_score(val0) / 10 as f32) + (card_val_score(val1) / 1000 as f32)
        }
        2 if counts == vec![4, 1] => {
          // FourKind
          let (_, val0) = histogram[0];
          let (_, val1) = histogram[1];

          0.8 + (card_val_score(val0) / 10 as f32) + (card_val_score(val1) / 1000 as f32)
        }
        _ => {
          let flush: bool = hand5.iter().all(|&card| card.suit == hand5[0].suit);
          let straight: bool = (hand5[4].value as usize - hand5[0].value as usize) == 4;

          if !straight && !flush {
            // HighCard
            values.reverse();
            0.1 + card_vals_score(values, 2)
          } else if straight && !flush {
            // Straight
            println!("{:?}", hand5);
            0.5 + (card_val_score(values[4]) / 10 as f32)
          } else if flush && !straight {
            // Flush
            values.reverse();
            0.8 + card_vals_score(values, 1)
          } else {
            if hand5[4].value != CardVal::CA {
              // StraightFlush
              0.9 + (card_val_score(values[4]) / 10 as f32)
            } else {
              // RoyalFlush
              1.0
            }
          }
        }
      }
    }
  }
}

pub trait Eval: Sync {
  fn score(&self, hand: &[Card]) -> Score;
}

impl Eval {
  pub fn naive() -> EvalNaive {
    EvalNaive {}
  }

  pub fn texas() -> EvalTexas {
    EvalTexas { combinations: utils::math::combinations(7, 5) }
  }

  pub fn texas_cache(path_synthetic: &Path) -> EvalTexasCache {
    let hands7: Vec<[Card; 7]> = {
      let ref path = path_synthetic.join("hands7.data");
      println!("loading {:?} ...", path);
      utils::bincode::deserialize_from_file_with_capacity(path, 1024 * 1024 * 128).unwrap()
    };

    let hands7_scores: Vec<f32> = {
      let ref path = path_synthetic.join("hands7_scores.data");
      println!("loading {:?} ...", path);
      utils::bincode::deserialize_from_file_with_capacity(path, 1024 * 1024 * 128).unwrap()
    };

    EvalTexasCache {
      hands7: Arc::new(Box::new(hands7)),
      hands7_scores: Arc::new(Box::new(hands7_scores)),
    }
  }
}

/** Naive evaluator compatible with Kuhn and Leduc. * * * * * * * * * * * *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

#[derive(Clone)]
pub struct EvalNaive {}

impl Eval for EvalNaive {
  fn score(&self, hand: &[Card]) -> Score {
    use anna_model::cards::{CARD_VALS, SUITS};

    let mut xs = vec![0; CARD_VALS.len()];

    for card in hand {
      xs[CARD_VALS.binary_search(&card.value).unwrap()] += 1;
    }

    let mut n = 0;
    let mut i = 0;

    for (j, &x) in xs.iter().enumerate() {
      if x > n {
        n = x;
        i = j;
      }
    }

    (i * n) as f32 / (CARD_VALS.len() * SUITS.len()) as f32
  }
}

#[derive(Clone)]
pub struct EvalTexas {
  combinations: Vec<Vec<usize>>,
}

//pub fn combinations(n: usize, length: usize) -> Vec<Vec<usize>> {
impl Eval for EvalTexas {
  // http://nsayer.blogspot.com/2007/07/algorithm-for-evaluating-domino-hands.html
  fn score(&self, hand7: &[Card]) -> Score {
    assert!(hand7.len() == 7);

    let mut score_max = 0.0;

    for c in self.combinations.iter() {
      let mut hand5 = Vec::new();
      for i in 0..5 {
        hand5.push(hand7[c[i]])
      }
      hand5.sort();

      let score = hand5_eval(&hand5);

      if score > score_max {
        score_max = score;
      }
    }

    score_max
  }
}

#[derive(Clone)]
pub struct EvalTexasCache {
  hands7: Arc<Box<Vec<[Card; 7]>>>,
  hands7_scores: Arc<Box<Vec<f32>>>,
}

impl Eval for EvalTexasCache {
  fn score(&self, xs: &[Card]) -> Score {
    assert!(xs.len() == 7);

    let mut hand7 = [xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6]];
    hand7.sort();

    let idx =
      self.hands7.binary_search(&hand7).expect(&format!("Hand not found in cache. ({:?})", hand7));
    self.hands7_scores[idx]
  }
}

/**
 ** Hand score between 0.0 and 1.0.
 **
 **/
pub type Score = f32;
