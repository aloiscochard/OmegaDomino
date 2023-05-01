#![feature(test)]
extern crate anna_eval;
extern crate anna_model;
extern crate rand;

#[test]
fn kuhn_eval() {
  use anna_eval::Eval;
  use anna_model::cards::KUHN_CARDS;

  let eval = Eval::naive();

  let s0 = eval.score(vec![KUHN_CARDS[0]]);
  let s1 = eval.score(vec![KUHN_CARDS[1]]);
  assert!(s1 > s0);
  let s2 = eval.score(vec![KUHN_CARDS[2]]);
  assert!(s2 > s1);
}

#[test]
fn leduc_eval() {
  use anna_eval::Eval;
  use anna_model::cards::*;

  let eval = Eval::naive();

  // OnePair
  let s0 = eval.score(vec![
    Card { suit: Suit::Spade, value: CardVal::CK },
    Card { suit: Suit::Heart, value: CardVal::CK },
  ]);

  let s1 = eval.score(vec![
    Card { suit: Suit::Spade, value: CardVal::CQ },
    Card { suit: Suit::Heart, value: CardVal::CQ },
  ]);

  assert!(s1 < s0);

  let s2 = eval.score(vec![
    Card { suit: Suit::Spade, value: CardVal::CJ },
    Card { suit: Suit::Heart, value: CardVal::CJ },
  ]);

  assert!(s2 < s1);

  // HighCard
  let s3 = eval.score(vec![
    Card { suit: Suit::Spade, value: CardVal::CQ },
    Card { suit: Suit::Heart, value: CardVal::CK },
  ]);

  assert!(s3 < s2);

  let s4 = eval.score(vec![
    Card { suit: Suit::Spade, value: CardVal::CJ },
    Card { suit: Suit::Heart, value: CardVal::CQ },
  ]);

  assert!(s4 < s3);
}

#[test]
fn leduc_strength() {
  use anna_eval::{strength, Eval};
  use anna_model::{cards::*, profile::profile_leduc};

  let ref mut rng = rand::thread_rng();

  let ref eval = Eval::naive();
  let ref profile = profile_leduc(2);
  let accuracy = 64;

  let score = {
    let hand = vec![Card { suit: Suit::Spade, value: CardVal::CJ }];
    let table = Vec::new();

    strength(rng, eval, profile, accuracy, &hand, &table)
  };

  assert!(score >= 0.2);

  let score = {
    let hand = vec![Card { suit: Suit::Spade, value: CardVal::CQ }];
    let table = Vec::new();

    strength(rng, eval, profile, accuracy, &hand, &table)
  };

  assert!(score >= 0.3);

  let score = {
    let hand = vec![Card { suit: Suit::Spade, value: CardVal::CK }];
    let table = Vec::new();

    strength(rng, eval, profile, accuracy, &hand, &table)
  };

  assert!(score >= 0.5);

  let score = {
    let hand = vec![Card { suit: Suit::Spade, value: CardVal::CK }];
    let table = vec![Card { suit: Suit::Heart, value: CardVal::CK }];

    strength(rng, eval, profile, accuracy, &hand, &table)
  };

  assert!(score == 1.0);
}
