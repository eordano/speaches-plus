use anyhow::Result;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    use candle_core::Device;
    use nv_runner::GreedyRunner;

    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<String> = None;
    let mut prompt: String = "The quick brown fox".to_string();
    let mut max_new: usize = 32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_dir = args.next(),
            "--prompt" => prompt = args.next().unwrap_or(prompt),
            "--max-new-tokens" => {
                max_new = args
                    .next()
                    .unwrap_or_else(|| "32".into())
                    .parse()
                    .unwrap_or(32);
            }
            _ => {}
        }
    }
    let model_dir = model_dir.ok_or_else(|| anyhow::anyhow!("--model <dir> required"))?;
    let device = Device::new_cuda(0)?;
    let mut runner = GreedyRunner::from_pretrained(std::path::Path::new(&model_dir), &device)?;
    let result = runner.generate(&prompt, max_new)?;
    println!("prompt_tokens: {:?}", result.prompt_tokens);
    println!("output_tokens: {:?}", result.output_tokens);
    println!("finish_reason: {:?}", result.finish_reason);
    println!("output_text: {}", result.output_text);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("qwen3_greedy example requires --features cuda")
}
