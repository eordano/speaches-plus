#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_mm_kind_matrix_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_mm_kind_matrix compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
         Re-run with --features cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use speaches_plus::oapi::chat_engine_wgpu::mm::{
        embed_row_route, media_present, refuse_media, MmCaps, MmRoute,
    };
    use speaches_plus::oapi::chat_engine_wgpu::WgpuModelKind;
    use speaches_plus::oapi::chat_multimodal::MmMedia;

    const ALL_KINDS: [WgpuModelKind; 7] = [
        WgpuModelKind::Gemma4Dense,
        WgpuModelKind::Gemma4E4b,
        WgpuModelKind::Gemma4Moe,
        WgpuModelKind::Qwen3_5Moe,
        WgpuModelKind::Qwen3_5Dense,
        WgpuModelKind::GptOss,
        WgpuModelKind::Laguna,
    ];

    fn kind_word(kind: WgpuModelKind) -> &'static str {
        kind.label()
            .split_whitespace()
            .next()
            .expect("every kind label starts with the kind name")
    }

    fn both_towers() -> MmCaps {
        MmCaps {
            images: true,
            audio: true,
        }
    }

    #[test]
    fn every_wgpu_kind_either_takes_embed_rows_or_refuses_with_its_own_name() {
        let mut serves = 0usize;
        let mut refuses = 0usize;
        for kind in ALL_KINDS {
            match embed_row_route(kind) {
                Ok(_) => {
                    serves += 1;
                    assert!(
                        refuse_media(kind, Some(both_towers()), true, true).is_none(),
                        "{kind:?} routes embed rows yet refuses a fully towered request"
                    );
                }
                Err(reason) => {
                    refuses += 1;
                    assert!(
                        reason.contains(kind_word(kind)),
                        "{kind:?} refusal must name the kind: {reason}"
                    );
                    assert!(
                        reason.len() > 80,
                        "{kind:?} refusal must say why, not just no: {reason}"
                    );
                }
            }
        }
        assert_eq!(
            serves + refuses,
            ALL_KINDS.len(),
            "every kind must land in exactly one column"
        );
        assert_eq!(
            serves, 3,
            "gemma4-dense, gemma4-moe and qwen3.5-dense pair an embed-row prefill entry with a \
             tower this tree can load"
        );
        assert_eq!(
            refuses, 4,
            "gemma4-e4b and qwen3.5-moe are missing one half each, laguna and gpt-oss have no \
             media checkpoint at all"
        );
    }

    #[test]
    fn the_gemma4_family_that_can_splice_routes_through_the_gemma4_towers() {
        assert_eq!(
            embed_row_route(WgpuModelKind::Gemma4Dense),
            Ok(MmRoute::Gemma4Towers)
        );
        assert_eq!(
            embed_row_route(WgpuModelKind::Gemma4Moe),
            Ok(MmRoute::Gemma4Towers)
        );
        assert_eq!(
            embed_row_route(WgpuModelKind::Qwen3_5Dense),
            Ok(MmRoute::Qwen3Vision)
        );
    }

    #[test]
    fn the_e4b_refusal_says_the_towers_exist_and_the_decoder_entry_does_not() {
        let reason = embed_row_route(WgpuModelKind::Gemma4E4b)
            .expect_err("e4b has no embed-row prefill entry in this tree");
        assert!(reason.contains("gemma4-e4b"), "{reason}");
        assert!(reason.contains("audio"), "{reason}");
        assert!(
            reason.contains("prefill_tokens_with_embed_rows"),
            "the refusal must name the missing decoder entry: {reason}"
        );
    }

    #[test]
    fn the_gpt_oss_refusal_says_the_decoder_takes_rows_and_the_checkpoint_ships_no_tower() {
        let reason = embed_row_route(WgpuModelKind::GptOss)
            .expect_err("no gpt-oss checkpoint ships a vision or audio config");
        assert!(reason.contains("gpt-oss"), "{reason}");
        assert!(
            reason.contains("prefill_tokens_with_image_rows"),
            "the refusal must credit the decoder half that exists: {reason}"
        );
        assert!(
            reason.contains("vision_config"),
            "the refusal must name the missing half as the checkpoint/tower: {reason}"
        );
    }

    #[test]
    fn an_audio_only_request_counts_as_media_so_it_can_never_reach_a_text_only_prefill() {
        let audio_only = MmMedia {
            images: Vec::new(),
            audios: vec![vec![0.0f32; 16000]],
        };
        assert!(
            media_present(&audio_only),
            "an audio-only request must be seen as multimodal; the seam once filtered on images \
             alone and served the prompt with the audio marker left in it"
        );
        let image_only = MmMedia {
            images: vec![image::RgbImage::new(4, 4)],
            audios: Vec::new(),
        };
        assert!(media_present(&image_only));
        assert!(!media_present(&MmMedia::default()));

        let images_only_tower = MmCaps {
            images: true,
            audio: false,
        };
        for kind in ALL_KINDS {
            let refusal = refuse_media(kind, Some(images_only_tower), false, true)
                .unwrap_or_else(|| panic!("{kind:?} accepted audio with no audio tower"));
            assert!(
                refusal.contains(kind_word(kind)),
                "{kind:?} audio refusal must name the kind: {refusal}"
            );
        }
    }

    #[test]
    fn a_missing_tower_refuses_rather_than_serving_a_marker_only_prompt() {
        for kind in ALL_KINDS {
            let refusal = refuse_media(kind, None, true, false)
                .unwrap_or_else(|| panic!("{kind:?} accepted an image with no runtime loaded"));
            assert!(
                refusal.contains(kind_word(kind)),
                "{kind:?} refusal must name the kind: {refusal}"
            );
        }
    }

    #[test]
    fn text_only_requests_stay_unrefused_on_every_kind() {
        for kind in ALL_KINDS {
            assert!(refuse_media(kind, None, false, false).is_none());
            assert!(refuse_media(kind, Some(both_towers()), false, false).is_none());
        }
    }
}
