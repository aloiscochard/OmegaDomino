use anna_model::{
  cards::{Card, CardVal},
  money::Money,
  profile::Profile,
};
use rand::{Rng, StdRng};

use anna_model::encoders::CardsEncoder;
use Eval;
use Score;

pub fn return_rate(hand_strength: Score, player_pots: &[Money], raise: Money) -> f32 {
  use std::cmp::max;
  let pot: u32 = player_pots.iter().map(|m| m.unpack()).sum();
  let pot_odds = max(1, raise.unpack()) as f32 / (raise.unpack() + pot) as f32;
  hand_strength / pot_odds
}

pub fn strength<R: Rng>(
  rng: &mut R,
  eval: &Eval,
  profile: &Profile,
  accuracy: usize,
  hand: &Vec<Card>,
  table: &Vec<Card>,
) -> Score {
  strength_(rng, eval, &profile.rounds, profile.players, &profile.deck, accuracy, hand, table)
}

/** Calculate hand strength with a given accuracy (number of MC
 ** simulations). ref: http://cowboyprogramming.com/2007/01/04/programming-domino-ai/
 **/
pub fn strength_<R: Rng>(
  rng: &mut R,
  eval: &Eval,
  rounds: &[usize],
  players: usize,
  deck_full: &[Card],
  accuracy: usize,
  hand: &Vec<Card>,
  table: &Vec<Card>,
) -> Score {
  use std::cmp::Ordering;

  let mut score = 0.0;

  let table_size: usize = rounds.iter().skip(1).sum();

  // Remove the known cards (your hole cards, and any community cards).
  let deck: Vec<&Card> =
    deck_full.iter().filter(|card| !hand.contains(&card) && !table.contains(&card)).collect();

  for _ in 0..accuracy {
    let mut deck: Vec<Card> = deck.iter().map(|&c| *c).collect();
    rng.shuffle(&mut deck);

    // Deal your opponents' hole cards, and the remaining community cards.
    let opponents: Vec<Vec<Card>> =
      (0..players - 1).map(|_| (0..rounds[0]).map(|_| deck.pop().unwrap()).collect()).collect();

    let mut table = table.clone();
    while table.len() < table_size {
      table.push(deck.pop().unwrap());
    }

    // Evaluate all hands
    let scores: Vec<f32> = (0..players)
      .map(|player_id| {
        let mut cards =
          if player_id == 0 { hand.clone() } else { opponents[player_id - 1].clone() };
        cards.extend(table.iter());
        eval.score(&cards)
      })
      .collect();

    // If have the best hand then add 1/(number of people with the same hand value)
    // to your score.
    let max = *scores.iter().max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap();
    if scores[0] == max {
      let winners = scores.iter().filter(|&s| *s == max).count();
      score += 1.0 / winners as f32;
    }
  }

  score / accuracy as f32
}

#[derive(Clone)]
pub struct CardsStrength<'a, E: Eval + Send + Sync> {
  accuracy: usize,
  eval: &'a E,
  profile: Profile,
  rng: StdRng,
}

impl<'a, E: Eval + Send + Sync> CardsStrength<'a, E> {
  pub fn new(rng: StdRng, eval: &'a E, profile: Profile, accuracy: usize) -> CardsStrength<'a, E> {
    CardsStrength { accuracy: accuracy, eval: eval, profile: profile, rng }
  }
}

impl<'a, E: Eval + Send + Sync> CardsEncoder for CardsStrength<'a, E> {
  fn encode(&mut self, xss: &Vec<Vec<Card>>, player_actives: usize) -> Vec<f32> {
    let hand = &xss[0];
    let table = xss[1..].iter().flatten().cloned().collect();
    let score = strength_(
      &mut self.rng,
      self.eval,
      &self.profile.rounds,
      player_actives,
      &self.profile.deck,
      self.accuracy,
      hand,
      &table,
    );

    vec![score]
  }

  fn size(&self) -> usize {
    1
  }
}
