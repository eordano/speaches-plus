    let ebase = slot * q3q_p.x_slot_stride_elems + kb * NVFP4_BLOCK_SIZE;
    var v0 = 0u;
    var v1 = 0u;
    if (live) {
        let gw = q3sq_g[(ebase >> 1u) + el];
        let uw = q3sq_u[((ebase + q3sq_p.u_off_elems) >> 1u) + el];
        let g0 = bf16_lo(gw);
        let g1 = bf16_hi(gw);
        let a0 = bf16_decode(bf16_encode(g0 / (1.0 + exp(-g0)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g1 / (1.0 + exp(-g1)))) * bf16_hi(uw);
        v0 = bf16_encode(a0) & 0xffffu;
        v1 = bf16_encode(a1) & 0xffffu;
    }
