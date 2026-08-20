package tts

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strconv"
)

type ResponseFormat string

const (
	FormatPCM  ResponseFormat = "pcm"
	FormatMP3  ResponseFormat = "mp3"
	FormatWAV  ResponseFormat = "wav"
	FormatFLAC ResponseFormat = "flac"
	FormatOPUS ResponseFormat = "opus"
	FormatAAC  ResponseFormat = "aac"
)

func MimeTypeForFormat(f ResponseFormat) string {
	switch f {
	case FormatPCM:
		return "audio/pcm"
	case FormatMP3:
		return "audio/mpeg"
	case FormatWAV:
		return "audio/wav"
	case FormatFLAC:
		return "audio/flac"
	case FormatOPUS:
		return "audio/opus"
	case FormatAAC:
		return "audio/aac"
	default:
		return "application/octet-stream"
	}
}

func ValidResponseFormat(f string) bool {
	switch ResponseFormat(f) {
	case FormatPCM, FormatMP3, FormatWAV, FormatFLAC, FormatOPUS, FormatAAC:
		return true
	default:
		return false
	}
}

func F32ToS16LE(samples []float32) []byte {
	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		v := s * 32767.0
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(math.Round(float64(v)))))
	}
	return out
}

func EncodeAudio(
	ctx context.Context,
	dst io.Writer,
	chunks <-chan []float32,
	sourceSR int,
	targetSR int,
	format ResponseFormat,
) error {
	if format == FormatPCM {
		for ch := range chunks {
			if _, err := dst.Write(F32ToS16LE(ch)); err != nil {
				return err
			}
		}
		return nil
	}

	if targetSR == 0 {
		targetSR = sourceSR
	}

	args := ffmpegArgsFor(format, sourceSR, targetSR)
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("ffmpeg stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("ffmpeg stdout: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("ffmpeg start: %w", err)
	}

	writeErrCh := make(chan error, 1)
	go func() {
		defer stdin.Close()
		var werr error
		for ch := range chunks {
			if _, e := stdin.Write(F32ToS16LE(ch)); e != nil {
				werr = e
				break
			}
		}
		writeErrCh <- werr
	}()

	_, copyErr := io.Copy(dst, stdout)
	waitErr := cmd.Wait()
	wErr := <-writeErrCh

	switch {
	case copyErr != nil:
		return fmt.Errorf("ffmpeg->client copy: %w", copyErr)
	case wErr != nil && !errors.Is(wErr, io.ErrClosedPipe):
		return fmt.Errorf("client->ffmpeg write: %w", wErr)
	case waitErr != nil:
		return fmt.Errorf("ffmpeg exit: %w", waitErr)
	}
	return nil
}

func ffmpegArgsFor(format ResponseFormat, sourceSR, targetSR int) []string {
	args := []string{
		"-f", "s16le",
		"-ar", strconv.Itoa(sourceSR),
		"-ac", "1",
		"-i", "pipe:0",
		"-ar", strconv.Itoa(targetSR),
	}
	switch format {
	case FormatMP3:
		args = append(args, "-f", "mp3", "-codec:a", "libmp3lame")
	case FormatWAV:
		args = append(args, "-f", "wav")
	case FormatFLAC:
		args = append(args, "-f", "flac")
	case FormatOPUS:
		args = append(args, "-f", "opus", "-codec:a", "libopus")
	case FormatAAC:
		args = append(args, "-f", "adts", "-codec:a", "aac")
	}
	args = append(args, "pipe:1", "-hide_banner", "-loglevel", "error")
	return args
}
