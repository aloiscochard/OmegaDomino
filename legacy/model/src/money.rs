extern crate serde;

use serde::{
  de::{Error, Visitor},
  Deserialize, Deserializer, Serialize, Serializer,
};

use std::{
  fmt,
  ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Money {
  value: u32,
}

impl Money {
  pub fn new(dollars: u16, cents: u8) -> Money {
    Money { value: (dollars as u32 * 100) + cents as u32 }
  }

  pub fn from_i32(value: i32) -> Option<Money> {
    if value >= 0 {
      Some(Money::from_u32(value as u32))
    } else {
      None
    }
  }

  pub fn from_u32(value: u32) -> Money {
    Money { value: value }
  }

  pub fn zero() -> Money {
    MONEY_ZERO
  }
  pub fn is_null(self) -> bool {
    self.value == 0
  }
  pub fn unpack(self) -> u32 {
    self.value
  }
}

impl Add for Money {
  type Output = Money;

  fn add(self, that: Money) -> Money {
    Money { value: self.value + that.value }
  }
}

impl Sub for Money {
  type Output = i32;
  fn sub(self, that: Money) -> i32 {
    self.value as i32 - that.value as i32
  }
}

impl Mul<u32> for Money {
  type Output = Money;
  fn mul(self, multiplier: u32) -> Money {
    Money { value: self.value * multiplier }
  }
}

impl Div<u32> for Money {
  type Output = Money;
  fn div(self, divisor: u32) -> Money {
    Money { value: self.value / divisor }
  }
}

impl<'de> Deserialize<'de> for Money {
  fn deserialize<D>(d: D) -> Result<Money, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct MoneyVisitor;

    impl<'de> Visitor<'de> for MoneyVisitor {
      type Value = Money;

      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer between 0 and u32.max")
      }

      fn visit_u32<E>(self, value: u32) -> Result<Money, E>
      where
        E: Error,
      {
        Ok(Money::from_u32(value))
      }
    }

    d.deserialize_u32(MoneyVisitor)
  }
}

impl fmt::Display for Money {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    if *self == MONEY_ZERO {
      f.write_str("0")?;
    } else {
      let s = format!("{:?}", self.value);
      match s.len() {
        1 => {
          f.write_str("0.0")?;
          f.write_str(&s)?;
        }
        2 => {
          f.write_str("0.")?;
          f.write_str(&s)?;
        }
        n => {
          f.write_str(&s[..n - 2])?;
          if &s[n - 2..] != "00" {
            f.write_str(".")?;
            f.write_str(&s[n - 2..])?;
          }
        }
      }
    }

    Ok(())
  }
}

impl Serialize for Money {
  fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    s.serialize_u32(self.value)
  }
}

pub const MONEY_ZERO: Money = Money { value: 0 };
