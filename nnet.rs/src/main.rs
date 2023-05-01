extern crate tensorflow;

extern crate nnet;

fn main() -> () {
  use nnet::{
    graph_load,
    tasks::{Eval, Train},
    NetworkContext, Tensor,
  };
  use std::io::{stdout, Write};

  // let (dataset_name, dataset_len) = ("t10k", 10000u32);
  let (dataset_name, dataset_len) = ("train", 60000u32);
  let hiddens = 3;

  let train_graph = graph_load("./nnets/mnist-train.pb").unwrap();
  let mut train = Train::new(NetworkContext::new(&train_graph, hiddens).unwrap());

  let eval_graph = graph_load("./nnets/mnist-eval.pb").unwrap();
  let mut eval = Eval::new(NetworkContext::new(&eval_graph, hiddens).unwrap());

  let is: Tensor<f32> = {
    let path = &format!("./datasets/mnist/{}-images-idx3", dataset_name);
    let bs = idx_load(vec![dataset_len, 28, 28], path).unwrap();
    let fs: Vec<f32> = bs.iter().map(|&x| f32::from(x) / 255.0).collect();
    Tensor::new(&[dataset_len as u64, 28, 28]).with_values(&fs).unwrap()
  };

  let os: Tensor<i32> = {
    let path = &format!("./datasets/mnist/{}-labels-idx1", dataset_name);
    let bs = idx_load(vec![dataset_len], path).unwrap();
    let us: Vec<i32> = bs.iter().map(|&x| x as i32).collect();
    Tensor::new(&[dataset_len as u64]).with_values(&us).unwrap()
  };

  use std::time::Instant;

  let training_start = Instant::now();

  for _ in 0..10 {
    let epoch_start = Instant::now();

    let mut scores = Vec::new();
    for _ in 0..16 {
      let score = train.run(0.001, &is, &os).unwrap();
      scores.push(score);

      let network = train.context.get().unwrap();
      eval.context.set(&network).unwrap();

      print!(".");
      let _ = stdout().flush();
    }
    println!("");

    let network = train.context.get().unwrap();
    eval.context.set(&network).unwrap();

    let score = eval.run(&is, &os).unwrap();
    println!("{:?} in {:?}", score, epoch_start.elapsed());
  }

  println!("");
  println!("in {:?}", training_start.elapsed());
}

fn idx_load(dims: Vec<u32>, path: &str) -> std::io::Result<Vec<u8>> {
  use std::{fs::File, io::Read};

  let mut file = File::open(path)?;
  let mut contents = Vec::new();

  file.read_to_end(&mut contents)?;

  for d in 1..(dims.len() + 1) {
    let bs = &contents[d * 4..(d + 1) * 4];
    let n = u32::from_be_bytes([bs[0], bs[1], bs[2], bs[3]]);
    assert!(n == dims[d - 1]);
  }

  let offset = (dims.len() + 1) * 4;

  Ok(contents[offset..].to_vec())
}
