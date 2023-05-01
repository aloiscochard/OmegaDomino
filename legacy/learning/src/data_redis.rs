use bincode::{deserialize, serialize, Infinite};
use rand::Rng;
use redis::{Client, Commands, Connection, RedisResult};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

pub struct RedisStore<A> {
  connection: Connection,
  namespace: Vec<u8>,
  value: PhantomData<A>,
}

impl<A> RedisStore<A> {
  pub fn new(client: &Client, namespace: Vec<u8>) -> RedisResult<RedisStore<A>> {
    let conn = client.get_connection()?;
    Ok(RedisStore { connection: conn, namespace: namespace, value: PhantomData })
  }

  pub fn key(&self, k: usize) -> Vec<u8> {
    let mut xs = self.namespace.clone();
    xs.extend(serialize(&k, Infinite).unwrap());
    xs
  }

  pub fn get(&mut self, k: usize) -> A
  where
    A: DeserializeOwned,
  {
    let xs: Vec<u8> = self.connection.get(self.key(k)).unwrap();
    deserialize(&xs).unwrap()
  }

  pub fn set(&mut self, k: usize, v: A) -> ()
  where
    A: Serialize,
  {
    let key = self.key(k);
    let xs = serialize(&v, Infinite).unwrap();
    let res: String = self.connection.set(key, xs).unwrap();
    assert!(res == "OK");
    ()
  }

  pub fn set_nx(&mut self, k: usize, v: A) -> ()
  where
    A: Serialize,
  {
    let key = self.key(k);
    let xs = serialize(&v, Infinite).unwrap();
    let res: u8 = self.connection.set_nx(key, xs).unwrap();

    match res {
      1 => self.connection.incr(self.namespace.clone(), 1).unwrap(),
      _ => panic!("Key exists."),
    }
  }

  pub fn field_get<T>(&mut self, label: &str) -> T
  where
    T: DeserializeOwned,
  {
    let key = self.field_key(label);
    let xs: Vec<u8> = self.connection.get(key).unwrap();
    deserialize(&xs).unwrap()
  }

  pub fn field_key(&self, label: &str) -> Vec<u8> {
    let mut xs = self.key(0);
    xs.extend(serialize(label, Infinite).unwrap());
    xs
  }

  pub fn field_set<T>(&mut self, label: &str, v: T) -> ()
  where
    T: Serialize,
  {
    let key = self.field_key(label);
    let xs: Vec<u8> = serialize(&v, Infinite).unwrap();
    let res: String = self.connection.set(key, xs).unwrap();
    assert!(res == "OK");
    ()
  }

  pub fn len(&mut self) -> usize {
    self.connection.get(self.namespace.clone()).unwrap_or(0)
  }
}

/** Circular Buffer
 **/
pub struct CBuffer<A> {
  pub size: usize,
  pub values: RedisStore<A>,
  i: usize,
  laps: usize,
}

impl<A> CBuffer<A> {
  pub fn new(size: usize, client: &Client, namespace: Vec<u8>) -> CBuffer<A> {
    CBuffer { size: size, values: RedisStore::new(client, namespace).unwrap(), i: 0, laps: 0 }
  }

  pub fn get(&mut self, k: usize) -> A
  where
    A: DeserializeOwned,
  {
    self.values.get(k)
  }

  pub fn push(&mut self, values: Vec<A>) -> ()
  where
    A: Serialize,
  {
    for value in values {
      let size = self.values.len();
      if size == self.size {
        if self.i == self.size {
          self.i = 0;
          self.laps += 1;
        }
      }

      if self.laps == 0 {
        self.values.set_nx(self.i, value);
      } else {
        self.values.set(self.i, value);
      }

      self.i += 1;
    }
  }

  pub fn usage(&mut self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }

  pub fn len(&mut self) -> usize {
    self.values.len()
  }

  pub fn snapshot_store(&mut self) -> () {
    self.values.field_set("i", self.i);
    self.values.field_set("laps", self.laps);
  }

  pub fn snapshot_load(&mut self) -> () {
    self.i = self.values.field_get("i");
    self.laps = self.values.field_get("laps");
  }
}

/** Reservoir Sampling (incremental), Algorithm R
 **  https://en.wikipedia.org/wiki/Reservoir_sampling
 **/

pub struct RSampler<A> {
  pub size: usize,
  pub values: RedisStore<A>,
  p_min: f32,
  i: usize,
}

impl<A> RSampler<A>
where
  A: Serialize,
{
  pub fn new(size: usize, p_min: f32, client: &Client, namespace: Vec<u8>) -> RSampler<A> {
    RSampler { size: size, values: RedisStore::new(client, namespace).unwrap(), p_min: p_min, i: 0 }
  }

  pub fn get(&mut self, k: usize) -> A
  where
    A: DeserializeOwned,
  {
    self.values.get(k)
  }

  pub fn push<R: Rng>(&mut self, rng: &mut R, values: Vec<A>) -> usize {
    use rand::distributions::{Distribution, Range};

    let mut count = 0;

    for value in values {
      let size = self.values.len();
      if size == self.size {
        let p = (1.0 / self.i as f32) * (self.size as f32);
        let p_norm = if p < self.p_min { self.p_min } else { p };

        if rng.gen::<f32>() < p_norm {
          count += 1;

          let range = Range::new(0, self.size);
          let j = range.sample(rng);

          self.values.set(j, value);
        }
      } else {
        count += 1;

        let i = self.i;
        self.values.set_nx(i, value);
      }
      self.i += 1;
    }

    count
  }

  pub fn usage(&mut self) -> f32 {
    let size = self.values.len();
    if size == self.size {
      1.0
    } else {
      size as f32 / self.size as f32
    }
  }

  pub fn len(&mut self) -> usize {
    self.values.len()
  }

  pub fn snapshot_store(&mut self) -> () {
    self.values.field_set("i", self.i);
  }

  pub fn snapshot_load(&mut self) -> () {
    self.i = self.values.field_get("i");
  }
}
