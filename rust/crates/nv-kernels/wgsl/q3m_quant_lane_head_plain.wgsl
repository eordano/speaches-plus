    let ebase = slot * q3q_p.x_slot_stride_elems + kb * NVFP4_BLOCK_SIZE;
    var v0 = 0u;
    var v1 = 0u;
    if (live) {
        let xw = q3q_x[(ebase >> 1u) + el];
        v0 = xw & 0xffffu;
        v1 = xw >> 16u;
    }
