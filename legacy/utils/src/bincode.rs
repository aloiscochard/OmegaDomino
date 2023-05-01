extern crate bincode;
extern crate serde;

use std::path::Path;

pub type Error = bincode::Error;
// TODO Contribute to bincode?

pub fn deserialize_from_file<A>(path: &Path) -> Result<A, bincode::Error>
where
  for<'de> A: serde::Deserialize<'de>,
{
  deserialize_from_file_(path, None)
}

pub fn deserialize_from_file_with_capacity<A>(
  path: &Path,
  capacity: usize,
) -> Result<A, bincode::Error>
where
  for<'de> A: serde::Deserialize<'de>,
{
  deserialize_from_file_(path, Some(capacity))
}

pub fn deserialize_from_file_<A>(path: &Path, capacity: Option<usize>) -> Result<A, bincode::Error>
where
  for<'de> A: serde::Deserialize<'de>,
{
  use std::{fs::File, io::BufReader};

  let ref file = File::open(path).expect(&format!("Unable to open {:?}", path));
  let mut reader =
    capacity.map(|c| BufReader::with_capacity(c, file)).unwrap_or(BufReader::new(file));
  bincode::deserialize_from(&mut reader, bincode::Infinite)
}

pub fn serialize_into_file<A>(path: &Path, value: &A) -> Result<(), bincode::Error>
where
  A: serde::Serialize,
{
  serialize_into_file_(path, value, None)
}

pub fn serialize_into_file_with_capacity<A>(
  path: &Path,
  value: &A,
  capacity: usize,
) -> Result<(), bincode::Error>
where
  A: serde::Serialize,
{
  serialize_into_file_(path, value, Some(capacity))
}

pub fn serialize_into_file_<A>(
  path: &Path,
  value: &A,
  capacity: Option<usize>,
) -> Result<(), bincode::Error>
where
  A: serde::Serialize,
{
  use std::{fs::File, io::BufWriter};

  let ref file = File::create(path).expect(&format!("Unable to create {:?}", path));
  let mut writer =
    capacity.map(|c| BufWriter::with_capacity(c, file)).unwrap_or(BufWriter::new(file));
  bincode::serialize_into(&mut writer, value, bincode::Infinite)
}
