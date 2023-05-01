extern crate chrono;
extern crate env_logger;
extern crate log;

use std::env;

pub fn init() {
  let format = |record: &log::LogRecord| {
    let now = chrono::UTC::now();
    format!(
      "{}.{:03}  - {} - {}",
      now.format("%Y-%m-%d %H:%M:%S"),
      now.timestamp_subsec_millis(),
      record.level(),
      record.args()
    )
  };

  let mut builder = env_logger::LogBuilder::new();
  builder.format(format);

  if env::var("RUST_LOG").is_ok() {
    builder.parse(&env::var("RUST_LOG").unwrap());
  }

  builder.init().unwrap();
}
