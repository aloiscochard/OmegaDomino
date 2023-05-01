use anna_model::{cards::Card, Money};
use players::Players;
use std::collections::HashSet;
use Event;

/*
pub fn players_setup<'a, CE: CardsEncoder>(players: &mut PlayersSel<Episode<'a, StdRng, CE>>,
                                           playing: HashSet<usize>)
                                           -> HashMap<usize, usize> {
  players.select(playing.len());

  let mut seat_id = 0;
  let mut seatsPerPlayer: HashMap<usize, usize> = HashMap::new();
  for p in 0..players.capacity() {
    if playing.contains(&p) {
      seatsPerPlayer.insert(p, seat_id);
      seat_id += 1;
    }
  }

  seatsPerPlayer
}
*/

/* Players Selector */
#[derive(Clone)]
pub struct PlayersSel<P> {
  pub players: Vec<P>,
  pub locations: Vec<usize>,
  players_i: usize,
}

impl<P> PlayersSel<P> {
  pub fn new(players: Vec<P>) -> PlayersSel<P> {
    assert!(players.len() >= 1);
    PlayersSel { players: players, players_i: 0, locations: Vec::new() }
  }

  pub fn capacity(&self) -> usize {
    self.players.len() + 1
  }

  pub fn player_at(&self, seat_id: usize) -> Option<usize> {
    self.locations.get(seat_id).cloned()
  }

  pub fn seat_of(&self, player_id: usize) -> Option<usize> {
    self.locations.iter().position(|&p| p == player_id)
  }

  pub fn setup(&mut self, playing: &HashSet<usize>) -> () {
    self.players_i = playing.len() - 2;
    assert!(self.players_i < self.players.len());

    self.locations = Vec::new();
    for i in 0..self.capacity() {
      if playing.contains(&i) {
        self.locations.push(i)
      }
    }
  }
}

impl<P, E> Players<E> for PlayersSel<P>
where
  P: Players<E>,
{
  fn init(
    &mut self,
    blinds: &[Money],
    player_first: usize,
    player_funds: &[Money],
    playing_hands: &Vec<(usize, Vec<Card>)>,
  ) -> () {
    self.players[self.players_i].init(blinds, player_first, player_funds, playing_hands)
  }

  fn play(
    &mut self,
    round_id: usize,
    table_target: Money,
    table_target_raise: Option<Money>,
    player_pots: &[Money],
    seat_id: usize,
    player_fund: Money,
    events: &[Event],
  ) -> Result<u8, E> {
    self.players[self.players_i].play(
      round_id,
      table_target,
      table_target_raise,
      player_pots,
      seat_id,
      player_fund,
      events,
    )
  }
}
