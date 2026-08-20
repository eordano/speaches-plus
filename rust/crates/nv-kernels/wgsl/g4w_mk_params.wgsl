
struct G4wMkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
};

@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;
