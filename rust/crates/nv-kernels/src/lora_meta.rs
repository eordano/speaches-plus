pub const NO_LORA: i32 = -1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraKernelMeta {
    pub max_loras: usize,
    pub token_lora_mapping: Vec<i32>,
    pub token_indices_sorted: Vec<i32>,
    pub num_tokens_per_lora: Vec<i32>,
    pub lora_token_start_loc: Vec<i32>,
    pub active_lora_ids: Vec<i32>,
    pub num_active_loras: usize,
    pub no_lora: bool,
}

impl LoraKernelMeta {
    pub fn prepare(token_lora_mapping: &[i32], max_loras: usize) -> Self {
        let t = token_lora_mapping.len();
        debug_assert!(token_lora_mapping
            .iter()
            .all(|&v| v >= -1 && v < max_loras as i32));

        let mut sorted: Vec<i32> = (0..t as i32).collect();
        sorted.sort_by_key(|&i| token_lora_mapping[i as usize]);

        let mut active_lora_ids = vec![NO_LORA; max_loras + 1];
        let mut num_tokens_per_lora = vec![0i32; max_loras + 1];
        let mut lora_token_start_loc = vec![0i32; max_loras + 2];

        let mut num_active = 0usize;
        let mut i = 0usize;
        while i < t {
            let id = token_lora_mapping[sorted[i] as usize];
            let mut j = i;
            while j < t && token_lora_mapping[sorted[j] as usize] == id {
                j += 1;
            }
            active_lora_ids[num_active] = id;
            num_tokens_per_lora[num_active] = (j - i) as i32;
            lora_token_start_loc[num_active + 1] =
                lora_token_start_loc[num_active] + num_tokens_per_lora[num_active];
            num_active += 1;
            i = j;
        }

        let no_lora = num_active == 0 || (num_active == 1 && active_lora_ids[0] == NO_LORA);

        Self {
            max_loras,
            token_lora_mapping: token_lora_mapping.to_vec(),
            token_indices_sorted: sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            num_active_loras: num_active,
            no_lora,
        }
    }

    pub fn grid_loras(&self) -> usize {
        self.max_loras + 1
    }
}
