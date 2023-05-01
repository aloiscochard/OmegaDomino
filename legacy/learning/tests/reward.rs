#![feature(test)]
extern crate anna_learning;
extern crate anna_model;

#[test]
fn reward_check() {
  use anna_learning::reward::*;
  use anna_model::Money;

  let blind = Money::new(1, 0);

  let r = RewardF { delta_biggest: blind * 200, discount_rate: 0.9 };

  let funds = vec![(blind * 100, blind * 90), (blind * 100, blind * 110)];

  let xs = r.observe(&funds, &vec![]);
  println!("{:?}", xs);
  assert!(xs[0] < 0.);
  assert!(xs[1] > 0.);

  let ys = r.discount(10.0, 4);
  println!("{:?}", ys);
}
