use nv_ocr::{BackendKind, OcrEngine};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: ocr_file <traineddata> <image...>");
    let engine = OcrEngine::from_traineddata(model.as_ref(), BackendKind::Classical).unwrap();
    for img in args {
        let bytes = std::fs::read(&img).unwrap();
        let t = std::time::Instant::now();
        match engine.recognize(&bytes) {
            Ok(res) => println!(
                "=== {img} ({} ms, {} tokens)\n{}",
                t.elapsed().as_millis(),
                res.tokens.len(),
                res.text
            ),
            Err(e) => println!("=== {img} ERROR: {e}"),
        }
    }
}
