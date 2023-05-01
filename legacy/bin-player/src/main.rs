extern crate anna_player;
extern crate anna_utils;

use std::path::Path;

use anna_player::{console, ps};
use anna_utils::logging;

pub fn main() {
  logging::init();
  let path_data_str = std::env::args().nth(1).unwrap();
  let ref path_data = Path::new(path_data_str.as_str());
  let mode = self::std::env::args().nth(2).unwrap();
  let task = self::std::env::args().nth(3);

  match mode.as_str() {
    "console" => {
      console::main(path_data, task);
    }
    "ps" => {
      ps::main(path_data, task);
    }
    _ => {
      println!("unknown mode: {}", mode);
    }
  }
}
