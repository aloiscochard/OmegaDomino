use tensorflow::{SessionRunArgs, TensorType};

use crate::{Error, NetworkContext, Tensor};

pub struct Eval<'a, I, O> {
  pub context: NetworkContext<'a, I, O>,
}

impl<'a, I, O> Eval<'a, I, O>
where
  I: TensorType,
  O: TensorType,
{
  pub fn new(context: NetworkContext<'a, I, O>) -> Eval<'a, I, O> {
    Eval { context: context }
  }

  pub fn run(&mut self, input: &Tensor<I>, target: &Tensor<O>) -> Result<f32, Error> {
    let ref mut op = self.context.op;

    let input_op = op.graph.operation_by_name_required("nnet_input")?;
    let target_op = op.graph.operation_by_name_required("nnet_target")?;
    let accuracy_op = op.graph.operation_by_name_required("nnet_accuracy")?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&input_op, 0, &input);
    ctx.add_feed(&target_op, 0, &target);

    let token = ctx.request_fetch(&accuracy_op, 0);
    op.session.run(&mut ctx)?;

    let result: Tensor<f32> = ctx.fetch(token)?;
    Ok(result[0])
  }
}

pub struct Train<'a, I, O> {
  pub context: NetworkContext<'a, I, O>,
}

impl<'a, I, O> Train<'a, I, O>
where
  I: TensorType,
  O: TensorType,
{
  pub fn new(context: NetworkContext<'a, I, O>) -> Train<'a, I, O> {
    Train { context: context }
  }

  pub fn predict(&mut self, input: &Tensor<I>) -> Result<Tensor<O>, Error> {
    use crate::tensor_of;

    let ref mut op = self.context.op;

    let learning_rate_t = tensor_of(&[], &[0.0f32])?;
    let learning_rate_op = op.graph.operation_by_name_required("nnet_learning_rate")?;

    let dropout_rate_t = tensor_of(&[], &[0.0f32])?;
    let dropout_rate_op = op.graph.operation_by_name_required("nnet_dropout_rate")?;

    let input_op = op.graph.operation_by_name_required("nnet_input")?;
    let output_op = op.graph.operation_by_name_required("nnet_output")?;

    let mut ctx = SessionRunArgs::new();

    ctx.add_feed(&learning_rate_op, 0, &learning_rate_t);
    ctx.add_feed(&dropout_rate_op, 0, &dropout_rate_t);
    ctx.add_feed(&input_op, 0, &input);

    let token = ctx.request_fetch(&output_op, 0);
    op.session.run(&mut ctx)?;

    Ok(ctx.fetch(token)?)
  }

  pub fn run_many(
    &mut self,
    times: usize,
    learning_rate: f32,
    input: &Tensor<I>,
    target: &Tensor<O>,
  ) -> Result<f32, Error> {
    let mut cost = 0.0;
    for _ in 0..times {
      cost += self.run(learning_rate, input, target)? / times as f32;
    }
    Ok(cost)
  }

  pub fn run_(
    &mut self,
    input: &Tensor<I>,
    target: &Tensor<O>,
    feeds: &[(String, f32)],
    fetches: &[String],
  ) -> Result<Vec<f32>, Error> {
    use crate::tensor_of;

    let ref mut op = self.context.op;

    let mut feeds_tops = Vec::new();
    for (feed, value) in feeds.iter() {
      let t = tensor_of(&[], &[*value])?;
      let op = op.graph.operation_by_name_required(&feed)?;
      feeds_tops.push((t, op))
    }

    let mut fetches_ops = Vec::new();
    for fetch in fetches {
      fetches_ops.push(op.graph.operation_by_name_required(fetch)?);
    }

    let input_op = op.graph.operation_by_name_required("nnet_input")?;
    let target_op = op.graph.operation_by_name_required("nnet_target")?;

    let train_op = op.graph.operation_by_name_required("train")?;

    let mut ctx = SessionRunArgs::new();

    for i in 0..feeds_tops.len() {
      ctx.add_feed(&feeds_tops[i].1, 0, &feeds_tops[i].0);
    }

    let mut fetches_tokens = Vec::new();
    for op in fetches_ops {
      fetches_tokens.push(ctx.request_fetch(&op, 0));
    }

    ctx.add_feed(&input_op, 0, &input);
    ctx.add_feed(&target_op, 0, &target);

    ctx.add_target(&train_op);

    op.session.run(&mut ctx)?;

    let mut results: Vec<f32> = Vec::new();
    for token in fetches_tokens {
      let result: Tensor<f32> = ctx.fetch(token)?;
      results.push(result[0])
    }

    Ok(results)
  }

  pub fn run(
    &mut self,
    learning_rate: f32,
    input: &Tensor<I>,
    target: &Tensor<O>,
  ) -> Result<f32, Error> {
    let results = self.run_(
      input,
      target,
      &[(String::from("nnet_learning_rate"), learning_rate)],
      &[String::from("nnet_cost")],
    )?;
    Ok(results[0])
  }
}

pub struct Predict<'a, I, O> {
  pub context: NetworkContext<'a, I, O>,
}

impl<'a, I, O> Predict<'a, I, O>
where
  I: TensorType,
  O: TensorType,
{
  pub fn new(context: NetworkContext<'a, I, O>) -> Predict<'a, I, O> {
    Predict { context: context }
  }

  pub fn run(&mut self, input: &Tensor<I>) -> Result<Tensor<O>, Error> {
    let ref mut op = self.context.op;

    let input_op = op.graph.operation_by_name_required("nnet_input")?;
    let output_op = op.graph.operation_by_name_required("nnet_output")?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&input_op, 0, &input);

    let token = ctx.request_fetch(&output_op, 0);
    op.session.run(&mut ctx)?;

    Ok(ctx.fetch(token)?)
  }

  pub fn run_get<T: TensorType>(
    &mut self,
    input: &Tensor<I>,
    output_name: &str,
  ) -> Result<Tensor<T>, Error> {
    let ref mut op = self.context.op;

    let input_op = op.graph.operation_by_name_required("nnet_input")?;
    let output_op = op.graph.operation_by_name_required(output_name)?;

    let mut ctx = SessionRunArgs::new();
    ctx.add_feed(&input_op, 0, &input);

    let token = ctx.request_fetch(&output_op, 0);
    op.session.run(&mut ctx)?;

    Ok(ctx.fetch(token)?)
  }
}
