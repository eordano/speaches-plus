package audio

/*
#cgo pkg-config: libavformat libavcodec libavutil libswresample
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <string.h>
#include <stdlib.h>

#define AVDECODE_OUT_RATE        16000
#define AVDECODE_AVIO_BUFSZ      4096
#define AVDECODE_PCM_INIT_CAP    16000

typedef struct {
    const uint8_t* data;
    int            size;
    int            pos;
} mem_buffer_t;

static int mem_read_packet(void* opaque, uint8_t* buf, int buf_size) {
    mem_buffer_t* mb = (mem_buffer_t*)opaque;
    int remaining = mb->size - mb->pos;
    if (remaining <= 0) return AVERROR_EOF;
    int n = remaining < buf_size ? remaining : buf_size;
    memcpy(buf, mb->data + mb->pos, n);
    mb->pos += n;
    return n;
}

static int64_t mem_seek(void* opaque, int64_t offset, int whence) {
    mem_buffer_t* mb = (mem_buffer_t*)opaque;
    int64_t target;
    if (whence == AVSEEK_SIZE) return mb->size;
    whence &= ~AVSEEK_FORCE;
    switch (whence) {
        case SEEK_SET: target = offset; break;
        case SEEK_CUR: target = mb->pos + offset; break;
        case SEEK_END: target = mb->size + offset; break;
        default: return -1;
    }
    if (target < 0 || target > mb->size) return -1;
    mb->pos = (int)target;
    return target;
}

static int av_decode_to_mono16k(
    const uint8_t* data, int size,
    float** out_pcm, int* out_samples
) {
    *out_pcm = NULL;
    *out_samples = 0;

    AVFormatContext* fmt = NULL;
    AVIOContext*     avio = NULL;
    uint8_t*         iobuf = NULL;
    AVCodecContext*  codec_ctx = NULL;
    SwrContext*      swr = NULL;
    AVPacket*        pkt = NULL;
    AVFrame*         frame = NULL;
    int              audio_stream = -1;
    int              ret = 0;
    float*           pcm = NULL;
    int              pcm_cap = 0, pcm_len = 0;

    mem_buffer_t* mb = (mem_buffer_t*)av_malloc(sizeof(mem_buffer_t));
    if (!mb) { ret = AVERROR(ENOMEM); goto done; }
    mb->data = data; mb->size = size; mb->pos = 0;

    iobuf = (uint8_t*)av_malloc(AVDECODE_AVIO_BUFSZ);
    if (!iobuf) { ret = AVERROR(ENOMEM); goto done; }

    avio = avio_alloc_context(iobuf, AVDECODE_AVIO_BUFSZ, 0, mb, mem_read_packet, NULL, mem_seek);
    if (!avio) { ret = AVERROR(ENOMEM); goto done; }

    fmt = avformat_alloc_context();
    if (!fmt) { ret = AVERROR(ENOMEM); goto done; }
    fmt->pb = avio;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    if ((ret = avformat_open_input(&fmt, NULL, NULL, NULL)) < 0) {
        // avformat_open_input frees fmt on failure.
        fmt = NULL;
        goto done;
    }
    if ((ret = avformat_find_stream_info(fmt, NULL)) < 0) goto done;

    for (unsigned int i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream = (int)i;
            break;
        }
    }
    if (audio_stream < 0) { ret = AVERROR_STREAM_NOT_FOUND; goto done; }

    AVCodecParameters* par = fmt->streams[audio_stream]->codecpar;
    const AVCodec* dec = avcodec_find_decoder(par->codec_id);
    if (!dec) { ret = AVERROR_DECODER_NOT_FOUND; goto done; }

    codec_ctx = avcodec_alloc_context3(dec);
    if (!codec_ctx) { ret = AVERROR(ENOMEM); goto done; }
    if ((ret = avcodec_parameters_to_context(codec_ctx, par)) < 0) goto done;
    if ((ret = avcodec_open2(codec_ctx, dec, NULL)) < 0) goto done;

    AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;
    AVChannelLayout in_ch_layout;
    if (codec_ctx->ch_layout.nb_channels > 0) {
        av_channel_layout_copy(&in_ch_layout, &codec_ctx->ch_layout);
    } else {
        av_channel_layout_default(&in_ch_layout, codec_ctx->ch_layout.nb_channels > 0 ? codec_ctx->ch_layout.nb_channels : 1);
    }

    swr = NULL;
    ret = swr_alloc_set_opts2(&swr,
        &out_ch_layout, AV_SAMPLE_FMT_FLT, AVDECODE_OUT_RATE,
        &in_ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate,
        0, NULL);
    av_channel_layout_uninit(&in_ch_layout);
    if (ret < 0 || !swr) { if (ret >= 0) ret = AVERROR(ENOMEM); goto done; }
    if ((ret = swr_init(swr)) < 0) goto done;

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) { ret = AVERROR(ENOMEM); goto done; }

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != audio_stream) {
            av_packet_unref(pkt);
            continue;
        }
        if ((ret = avcodec_send_packet(codec_ctx, pkt)) < 0) {
            av_packet_unref(pkt);
            goto done;
        }
        av_packet_unref(pkt);
        while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
            int max_out = av_rescale_rnd(
                swr_get_delay(swr, codec_ctx->sample_rate) + frame->nb_samples,
                AVDECODE_OUT_RATE, codec_ctx->sample_rate, AV_ROUND_UP);
            if (pcm_len + max_out > pcm_cap) {
                int new_cap = pcm_cap ? pcm_cap * 2 : AVDECODE_PCM_INIT_CAP;
                while (new_cap < pcm_len + max_out) new_cap *= 2;
                float* tmp = (float*)av_realloc(pcm, sizeof(float) * new_cap);
                if (!tmp) { ret = AVERROR(ENOMEM); goto done; }
                pcm = tmp; pcm_cap = new_cap;
            }
            uint8_t* out_ptrs[1] = { (uint8_t*)(pcm + pcm_len) };
            int converted = swr_convert(swr, out_ptrs, max_out,
                (const uint8_t**)frame->extended_data, frame->nb_samples);
            if (converted < 0) { ret = converted; goto done; }
            pcm_len += converted;
            av_frame_unref(frame);
        }
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ret = 0;
        else if (ret < 0) goto done;
    }

    avcodec_send_packet(codec_ctx, NULL);
    while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
        int max_out = av_rescale_rnd(
            swr_get_delay(swr, codec_ctx->sample_rate) + frame->nb_samples,
            AVDECODE_OUT_RATE, codec_ctx->sample_rate, AV_ROUND_UP);
        if (pcm_len + max_out > pcm_cap) {
            int new_cap = pcm_cap ? pcm_cap * 2 : AVDECODE_PCM_INIT_CAP;
            while (new_cap < pcm_len + max_out) new_cap *= 2;
            float* tmp = (float*)av_realloc(pcm, sizeof(float) * new_cap);
            if (!tmp) { ret = AVERROR(ENOMEM); goto done; }
            pcm = tmp; pcm_cap = new_cap;
        }
        uint8_t* out_ptrs[1] = { (uint8_t*)(pcm + pcm_len) };
        int converted = swr_convert(swr, out_ptrs, max_out,
            (const uint8_t**)frame->extended_data, frame->nb_samples);
        if (converted < 0) { ret = converted; goto done; }
        pcm_len += converted;
        av_frame_unref(frame);
    }
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ret = 0;

    {
        int max_out = swr_get_out_samples(swr, 0);
        if (max_out > 0) {
            if (pcm_len + max_out > pcm_cap) {
                int new_cap = pcm_len + max_out;
                float* tmp = (float*)av_realloc(pcm, sizeof(float) * new_cap);
                if (!tmp) { ret = AVERROR(ENOMEM); goto done; }
                pcm = tmp; pcm_cap = new_cap;
            }
            uint8_t* out_ptrs[1] = { (uint8_t*)(pcm + pcm_len) };
            int converted = swr_convert(swr, out_ptrs, max_out, NULL, 0);
            if (converted > 0) pcm_len += converted;
        }
    }

    *out_pcm = pcm;
    *out_samples = pcm_len;
    pcm = NULL;
    ret = 0;

done:
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (swr) swr_free(&swr);
    if (fmt) avformat_close_input(&fmt);
    if (avio) {
        av_freep(&avio->buffer);
        avio_context_free(&avio);
        iobuf = NULL;
    }
    if (iobuf) av_free(iobuf);
    if (mb) av_free(mb);
    if (pcm) av_free(pcm);
    return ret;
}

static const char* av_strerr(int code) {
    static __thread char buf[256];
    if (av_strerror(code, buf, sizeof(buf)) < 0) {
        snprintf(buf, sizeof(buf), "unknown av error %d", code);
    }
    return buf;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func DecodeAnyToMono16k(data []byte) (MonoF32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("avdecode: empty input")
	}
	var (
		outPCM     *C.float
		outSamples C.int
	)
	rc := C.av_decode_to_mono16k(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
		&outPCM,
		&outSamples,
	)
	if rc < 0 {
		return nil, fmt.Errorf("avdecode: %s", C.GoString(C.av_strerr(rc)))
	}
	defer C.free(unsafe.Pointer(outPCM))

	n := int(outSamples)
	out := make(MonoF32, n)
	if n > 0 {
		src := unsafe.Slice((*float32)(unsafe.Pointer(outPCM)), n)
		copy(out, src)
	}
	return out, nil
}
