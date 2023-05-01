use nnet::{self, tensor1, Tensor};

use anna_model::cards::Card;
use anna_simulation::{Act, Event, SeatId};
use qnet::{QNet, QState};

pub type Behavior = (QState, u8);

pub struct BehaviorSet {
  pub inputs: Tensor<f32>,
  pub outputs: Tensor<f32>,
  pub size: usize,
}

pub fn extract_all(
  qnet: &QNet,
  player: SeatId,
  player_cards: &Vec<Card>,
  events: &Vec<Event>,
) -> Vec<Behavior> {
  let mut state = QState::new(qnet, player, player_cards);

  let mut xs = Vec::new();
  for ev in events {
    match ev {
      &Event::Play(Act { seat_id, action }) if seat_id == player => {
        xs.push((state.clone(), action));
      }
      _ => {
        // NOOP
      }
    }

    state.update(qnet, ev);
  }
  xs
}
