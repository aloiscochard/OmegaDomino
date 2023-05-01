use anna_model::Money;

pub type Reward = f32; // (-1.0 / +1.0)

pub struct RewardF {
  pub delta_biggest: Money,
}

impl RewardF {
  pub fn new(delta_biggest: Money) -> RewardF {
    RewardF { delta_biggest: delta_biggest }
  }

  // Observe the (immediate) reward for each player of a given game using their
  // inital and terminal funds.
  pub fn observe(&self, funds: &[(Money, Money)], winners: &[usize]) -> Vec<Reward> {
    funds
      .iter()
      .enumerate()
      .map(|(i, &(fs_init, fs_final))| {
        let delta = fs_final - fs_init;
        // let delta_n = delta;
        let delta_n = if winners.contains(&i) { delta + 1 } else { delta - 1 };

        let r = delta_n as f32 / (self.delta_biggest.unpack() as f32 + 1.0);
        // println!("delta: {}", delta);
        // println!("delta_n: {} / {}", delta_n, self.delta_biggest.unpack());
        // println!("r: {}", r);

        assert!(r <= 1.0);
        assert!(r >= -1.0);
        assert!(r != 0.0);

        r
      })
      .collect()
  }
}
