#[derive(Clone)]
pub enum HParam {
  Const(f32),
  Decay(f32),              // Proportional to inverse sqrt of iterations.
  Linear(f32, f32, usize), // (start, finish, iterations)
}

impl HParam {
  pub fn new(x: f32) -> HParam {
    HParam::Const(x)
  }
  pub fn decay(x: f32) -> HParam {
    HParam::Decay(x)
  }
  pub fn linear(x: f32, y: f32, i: usize) -> HParam {
    HParam::Linear(x, y, i)
  }

  pub fn apply(&self, i: usize) -> f32 {
    use self::HParam::*;

    match self {
      Const(x) => *x,
      Decay(x) => x * (1.0 / (usize::max(1, i) as f32).sqrt()),
      Linear(x, y, n) => {
        if i >= *n {
          *y
        } else {
          let r = (n - i) as f32 / *n as f32;
          ((x - y) * r) + y
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use hparam::*;

  #[test]
  fn hparam_linear_dec() -> () {
    let hparam = HParam::linear(1.0, 0.0, 4);
    assert!(hparam.apply(5) == 0.0);
    assert!(hparam.apply(4) == 0.0);
    assert!(hparam.apply(3) == 0.25);
    assert!(hparam.apply(2) == 0.50);
    assert!(hparam.apply(1) == 0.75);
    assert!(hparam.apply(0) == 1.0);
  }

  #[test]
  fn hparam_linear_inc() -> () {
    let hparam = HParam::linear(0.0, 1.0, 4);
    assert!(hparam.apply(0) == 0.0);
    assert!(hparam.apply(1) == 0.25);
    assert!(hparam.apply(2) == 0.50);
    assert!(hparam.apply(3) == 0.75);
    assert!(hparam.apply(4) == 1.0);
    assert!(hparam.apply(5) == 1.0);
  }
}
