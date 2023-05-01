#[macro_use]
extern crate quickcheck;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate tensorflow;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{error, marker::PhantomData};
use tensorflow::{SessionRunArgs, TensorType};

pub mod data;
pub mod tasks;

pub type Error = Box<error::Error>;
pub type Graph = tensorflow::Graph;
pub type Session = tensorflow::Session;
pub type Tensor<A> = tensorflow::Tensor<A>;

pub fn tensor1<A: TensorType>(values: &[A]) -> Result<Tensor<A>, Error> {
  Ok(Tensor::new(&[1, values.len() as u64]).with_values(values)?)
}

pub fn tensor_of<A: TensorType>(dims: &[u64], values: &[A]) -> Result<Tensor<A>, Error> {
  Ok(Tensor::new(dims).with_values(values)?)
}

#[derive(Clone)]
pub struct TensorS<A: TensorType> {
  pub tensor: Tensor<A>,
}

impl<A> Serialize for TensorS<A>
where
  A: Serialize,
  A: TensorType,
{
  fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let xs: Vec<&A> = self.tensor.iter().collect();
    (self.tensor.dims(), xs).serialize(s)
  }
}

impl<'de, A> Deserialize<'de> for TensorS<A>
where
  A: TensorType,
  A: Deserialize<'de>,
{
  fn deserialize<D>(d: D) -> Result<TensorS<A>, D::Error>
  where
    D: Deserializer<'de>,
  {
    Deserialize::deserialize(d).and_then(|(dims, xs): (Vec<u64>, Vec<A>)| {
      Ok(TensorS {
        tensor: Tensor::new(&dims).with_values(&xs).map_err(|e| serde::de::Error::custom(e))?,
      })
    })
  }
}

pub fn graph_load(graph_path: &str) -> Result<Graph, Error> {
  use std::{fs::File, io::Read};
  use tensorflow::ImportGraphDefOptions;

  let mut g = Graph::new();
  let mut g_data = Vec::new();
  File::open(graph_path)?.read_to_end(&mut g_data)?;
  g.import_graph_def(&g_data, &ImportGraphDefOptions::new())?;

  Ok(g)
}

pub fn session<'a>(graph: &'a Graph) -> Result<TensorOp<'a>, Error> {
  use tensorflow::SessionOptions;

  Ok(TensorOp { session: Session::new(&SessionOptions::new(), &graph)?, graph: graph })
}

pub fn session_init<'a>(graph: &'a Graph) -> Result<TensorOp<'a>, Error> {
  let mut op = session(graph)?;

  let mut ctx = SessionRunArgs::new();
  ctx.add_target(&op.graph.operation_by_name_required("init")?);
  op.session.run(&mut ctx)?;

  Ok(op)
}

pub fn network_init(graph: &Graph, hidden_layers: usize) -> Result<Network, Error> {
  let mut ctx = NetworkContext::<f32, f32>::new(graph, hidden_layers)?;
  ctx.get()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
  pub biases: Vec<TensorS<f32>>,
  pub weights: Vec<TensorS<f32>>,
}

pub struct NetworkContext<'a, I, O> {
  pub hidden_layers: usize,
  pub op: TensorOp<'a>,
  pub input_type: PhantomData<I>,
  pub output_type: PhantomData<O>,
}

impl<'a, I, O> NetworkContext<'a, I, O>
where
  I: TensorType,
  O: TensorType,
{
  pub fn new(graph: &'a Graph, hidden_layers: usize) -> Result<NetworkContext<'a, I, O>, Error> {
    Ok(NetworkContext {
      hidden_layers: hidden_layers,
      op: session_init(graph)?,
      input_type: PhantomData,
      output_type: PhantomData,
    })
  }

  pub fn optimizer_reset(&mut self) -> Result<(), Error> {
    let mut ctx = SessionRunArgs::new();
    ctx.add_target(&self.op.graph.operation_by_name_required("nnet_optimizer_init")?);
    self.op.session.run(&mut ctx)?;
    Ok(())
  }

  pub fn set(&mut self, network: &Network) -> Result<(), Error> {
    let mut ctx = SessionRunArgs::new();

    for l in 0..network.weights.len() {
      let bs = self.op.graph.operation_by_name_required(&format!("nnet_bs_{}_init", l))?;
      let ws = self.op.graph.operation_by_name_required(&format!("nnet_ws_{}_init", l))?;

      ctx.add_feed(&bs, 0, &network.biases[l].tensor);
      ctx.add_feed(&ws, 0, &network.weights[l].tensor);
    }

    ctx.add_target(&self.op.graph.operation_by_name_required("nnet_init")?);

    self.op.session.run(&mut ctx)?;

    Ok(())
  }

  pub fn get(&mut self) -> Result<Network, Error> {
    let mut ctx = SessionRunArgs::new();
    let mut tokens = Vec::new();

    for l in 0..(self.hidden_layers + 1) {
      let bs = self.op.graph.operation_by_name_required(&format!("nnet_bs_{}", l))?;
      let ws = self.op.graph.operation_by_name_required(&format!("nnet_ws_{}", l))?;

      let bs_t = ctx.request_fetch(&bs, 0);
      let ws_t = ctx.request_fetch(&ws, 0);

      tokens.push((bs_t, ws_t));
    }

    self.op.session.run(&mut ctx)?;

    let mut bs = Vec::new();
    let mut ws = Vec::new();

    for (bs_t, ws_t) in tokens {
      bs.push(TensorS { tensor: ctx.fetch(bs_t)? });
      ws.push(TensorS { tensor: ctx.fetch(ws_t)? });
    }

    Ok(Network { biases: bs, weights: ws })
  }
}

pub struct TensorOp<'a> {
  pub graph: &'a Graph,
  pub session: Session,
}

unsafe impl<'a> Sync for TensorOp<'a> {}
unsafe impl<'a> Send for TensorOp<'a> {}

pub struct TensorSystem<'a> {
  pub sys_op: TensorOp<'a>,
}

impl<'a> TensorSystem<'a> {
  pub fn new(graph: &'a Graph) -> Result<TensorSystem<'a>, Error> {
    Ok(TensorSystem { sys_op: session(graph)? })
  }

  pub fn concat_all(&mut self, xss: &[Tensor<f32>]) -> Result<Tensor<f32>, Error> {
    if xss.len() > 1 {
      let mut xs = xss[0].clone();
      for i in 1..xss.len() {
        xs = self.concat(&xs, &xss[i])?;
      }
      Ok(xs)
    } else {
      Ok(xss[0].clone())
    }
  }

  pub fn concat(&mut self, xs: &Tensor<f32>, ys: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
    let ref mut top = self.sys_op;

    let xs_op = top.graph.operation_by_name_required("concat/xs")?;
    let ys_op = top.graph.operation_by_name_required("concat/ys")?;
    let zs_op = top.graph.operation_by_name_required("concat/zs")?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&xs_op, 0, &xs);
    ctx.add_feed(&ys_op, 0, &ys);
    let token = ctx.request_fetch(&zs_op, 0);
    top.session.run(&mut ctx)?;

    let result = ctx.fetch(token)?;
    Ok(result)
  }

  pub fn gather(&mut self, xs: &Tensor<f32>, is: &[i32]) -> Result<Tensor<f32>, Error> {
    let ref mut top = self.sys_op;

    let xs_op = top.graph.operation_by_name_required("gather/xs")?;
    let is_op = top.graph.operation_by_name_required("gather/indices")?;
    let zs_op = top.graph.operation_by_name_required("gather/zs")?;

    let is_t = tensorflow::Tensor::new(&[is.len() as u64]).with_values(is)?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&xs_op, 0, &xs);
    ctx.add_feed(&is_op, 0, &is_t);
    let token = ctx.request_fetch(&zs_op, 0);
    top.session.run(&mut ctx)?;

    let result = ctx.fetch(token)?;
    Ok(result)
  }

  pub fn reduce_max(&mut self, xs: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
    let ref mut top = self.sys_op;

    let xs_op = top.graph.operation_by_name_required("reduce_max/xs")?;
    let zs_op = top.graph.operation_by_name_required("reduce_max/zs")?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&xs_op, 0, &xs);
    let token = ctx.request_fetch(&zs_op, 0);
    top.session.run(&mut ctx)?;

    let result = ctx.fetch(token)?;
    Ok(result)
  }

  pub fn row_set(
    &mut self,
    xs: &Tensor<f32>,
    i: i32,
    x: &Tensor<f32>,
  ) -> Result<Tensor<f32>, Error> {
    let ref mut top = self.sys_op;

    let xs_op = top.graph.operation_by_name_required("row_set/xs")?;
    let i_op = top.graph.operation_by_name_required("row_set/i")?;
    let x_op = top.graph.operation_by_name_required("row_set/x")?;
    let zs_op = top.graph.operation_by_name_required("row_set/zs")?;

    let i_t = tensorflow::Tensor::new(&[]).with_values(&[i])?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&xs_op, 0, &xs);
    ctx.add_feed(&i_op, 0, &i_t);
    ctx.add_feed(&x_op, 0, &x);
    let token = ctx.request_fetch(&zs_op, 0);
    top.session.run(&mut ctx)?;

    let result = ctx.fetch(token)?;
    Ok(result)
  }
}

/*
#[cfg(test)]
mod benchs {
  use crate::test::Bencher;
  use crate::{Tensor, TensorSystem};

  #[bench]
  fn concat_session_recreate(bencher: &mut Bencher) {
    use crate::{graph_load, session, Graph};

    let graph_path = "./graphs/system.pb";

    use std::{fs::File, io::Read};
    use tensorflow::ImportGraphDefOptions;

    let mut g_data = Vec::new();
    File::open(graph_path).unwrap().read_to_end(&mut g_data).unwrap();

    let xs = Tensor::new(&[1]).with_values(&[0.]).unwrap();
    let ys = Tensor::new(&[1]).with_values(&[1.]).unwrap();

    bencher.iter(|| {
      let mut graph = Graph::new();
      graph.import_graph_def(&g_data, &ImportGraphDefOptions::new())?;

      let mut sys = TensorSystem { sys_op: session(graph).unwrap() };
      sys.concat(&xs, &ys)
    });
  }

  #[bench]
  fn concat_session_reuse(bencher: &mut Bencher) {
    let mut sys = TensorSystem::new().unwrap();
    let xs = Tensor::new(&[1]).with_values(&[0.]).unwrap();
    let ys = Tensor::new(&[1]).with_values(&[1.]).unwrap();

    bencher.iter(|| sys.concat(&xs, &ys));
  }
}
*/

#[cfg(test)]
mod tests {
  use crate::{Tensor, TensorSystem};
  use tensorflow::Graph;

  #[test]
  fn tf_devices_list() -> () {
    use tensorflow::{Session, SessionOptions};

    let graph = Graph::new();
    let session = Session::new(&SessionOptions::new(), &graph).unwrap();
    let devices = session.device_list().unwrap();

    assert!(devices.len() > 0);

    for d in devices {
      println!("{:?}", d);
    }
  }

  quickcheck! {
    fn concat(xs: Vec<f32>, ys: Vec<f32>) -> bool {
      let graph = Graph::new();
      let mut sys = TensorSystem::new(&graph).unwrap();
      let xs_t = Tensor::new(&[xs.len() as u64]).with_values(&xs).unwrap();
      let ys_t = Tensor::new(&[ys.len() as u64]).with_values(&ys).unwrap();
      let zs_t = sys.concat(&xs_t, &ys_t).unwrap();
      let mut zs = xs;
      zs.extend(ys);

      zs.iter().collect::<Vec<&f32>>() == zs_t.iter().collect::<Vec<&f32>>()
    }

    fn gather(xs: Vec<f32>, ys: Vec<u32>) -> bool {
      let graph = Graph::new();
      let mut sys = TensorSystem::new(&graph).unwrap();
      let is: Vec<i32> =
        ys.into_iter().filter(|&y| y < xs.len() as u32).map(|y| y as i32).collect();
      let xs_t = Tensor::new(&[xs.len() as u64]).with_values(&xs).unwrap();
      let zs_t = sys.gather(&xs_t, &is).unwrap();
      let zs: Vec<&f32> = is.iter().map(|&i| &xs[i as usize]).collect();

      zs == zs_t.iter().collect::<Vec<&f32>>()
    }

    fn reduce_max(xs: Vec<f32>) -> bool {
      use std::cmp::Ordering;

      if xs.len() < 1 { return true; }

      let graph = Graph::new();
      let mut sys = TensorSystem::new(&graph).unwrap();

      let xs_t = Tensor::new(&[xs.len() as u64]).with_values(&xs).unwrap();
      let zs_t = sys.reduce_max(&xs_t).unwrap();

      let zs: Vec<&f32> = vec![xs.iter().max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap()];
      zs == zs_t.iter().collect::<Vec<&f32>>()
    }

    fn row_set(xs: Vec<f32>, ys: Vec<u32>, x: f32) -> bool {
      let graph = Graph::new();
      let mut sys = TensorSystem::new(&graph).unwrap();
      if xs.len() < 2 { return true; }

      let is: Vec<i32> =
        ys.into_iter().filter(|&y| y < xs.len() as u32).map(|y| y as i32).collect();
      if is.len() < 1 { return true; }

      let i = is[0];

      let xs_t = Tensor::new(&[xs.len() as u64]).with_values(&xs).unwrap();
      let x_t = Tensor::new(&[1]).with_values(&[x]).unwrap();
      let zs_t = sys.row_set(&xs_t, i, &x_t).unwrap();
      let mut zs: Vec<&f32> = Vec::new();
      for i in 0..xs.len() { zs.push(&xs[i]); }
      zs[i as usize] = &x;

      zs == zs_t.iter().collect::<Vec<&f32>>()
    }
  }
}
