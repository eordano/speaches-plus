package inspect

import (
	"encoding/binary"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Channel string

func ChannelSampleRate(c Channel) int {
	switch c {
	case ChannelMicIn:
		return defaultMicSampleRate
	case ChannelTTSOut:
		return defaultTTSSampleRate
	}
	return 0
}

type track struct {
	mu           sync.Mutex
	path         string
	sampleRate   int
	file         *os.File
	totalSamples int64
	firstNS      int64
	startMonoNS  int64
}

func newTrack(sessionID string, channel Channel, sessionDir string, startMonoNS int64) *track {
	t := &track{
		sampleRate:  ChannelSampleRate(channel),
		startMonoNS: startMonoNS,
		path:        filepath.Join(sessionDir, sessionID+".audio_"+string(channel)+".raw"),
	}
	if err := os.MkdirAll(sessionDir, 0o755); err != nil {
		slog.Warn("audio_store: mkdir", "err", err)
		return t
	}
	f, err := os.OpenFile(t.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		slog.Warn("audio_store: open track", "err", err, "path", t.path)
		return t
	}
	t.file = f
	return t
}

func (t *track) appendPCM16(pcm []byte) {
	if len(pcm) == 0 || t.file == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.firstNS == 0 {
		t.firstNS = time.Now().UnixNano()
	}
	if _, err := t.file.Write(pcm); err != nil {
		slog.Warn("audio_store: write", "err", err)
		return
	}
	t.totalSamples += int64(len(pcm) / 2)
}

func (t *track) appendFloat32(samples []float32) {
	if len(samples) == 0 {
		return
	}
	pcm := make([]byte, len(samples)*2)
	for i, s := range samples {
		v := int32(s * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		pcm[2*i] = byte(v)
		pcm[2*i+1] = byte(v >> 8)
	}
	t.appendPCM16(pcm)
}

func (t *track) offsetMs() int64 {
	if t.firstNS == 0 {
		return 0
	}
	d := (t.firstNS - t.startMonoNS) / 1_000_000
	if d < 0 {
		return 0
	}
	return d
}

func (t *track) slice(fromMs, toMs int64) []byte {
	if t.file == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	_ = t.file.Sync()
	src, err := os.Open(t.path)
	if err != nil {
		slog.Warn("audio_store: read open", "err", err, "path", t.path)
		return nil
	}
	defer src.Close()
	st, err := src.Stat()
	if err != nil {
		return nil
	}
	size := st.Size()
	from := fromMs * int64(t.sampleRate) * 2 / 1000
	if from < 0 {
		from = 0
	}
	if from > size {
		return nil
	}
	to := size
	if toMs > 0 {
		to = toMs * int64(t.sampleRate) * 2 / 1000
		if to > size {
			to = size
		}
	}
	if to <= from {
		return nil
	}
	if _, err := src.Seek(from, 0); err != nil {
		return nil
	}
	out := make([]byte, to-from)
	n, err := src.Read(out)
	if err != nil && n == 0 {
		return nil
	}
	return out[:n]
}

func (t *track) close() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.file != nil {
		_ = t.file.Sync()
		_ = t.file.Close()
		t.file = nil
	}
}

type AudioStore struct {
	sessionID   string
	sessionDir  string
	startMonoNS int64
	startWall   float64
	tracks      map[Channel]*track
}

func NewAudioStore(sessionID, sessionDir string) *AudioStore {
	now := time.Now()
	startMono := now.UnixNano()
	a := &AudioStore{
		sessionID:   sessionID,
		sessionDir:  sessionDir,
		startMonoNS: startMono,
		startWall:   float64(now.UnixNano()) / 1e9,
		tracks: map[Channel]*track{
			ChannelMicIn:  newTrack(sessionID, ChannelMicIn, sessionDir, startMono),
			ChannelTTSOut: newTrack(sessionID, ChannelTTSOut, sessionDir, startMono),
		},
	}
	return a
}

func (a *AudioStore) AppendMicIn(samples []float32) {
	if a == nil {
		return
	}
	a.tracks[ChannelMicIn].appendFloat32(samples)
}

func (a *AudioStore) AppendTTSOut(pcm16 []byte) {
	if a == nil {
		return
	}
	a.tracks[ChannelTTSOut].appendPCM16(pcm16)
}

func (a *AudioStore) AppendTTSOutFloat32(samples []float32) {
	if a == nil {
		return
	}
	a.tracks[ChannelTTSOut].appendFloat32(samples)
}

func (a *AudioStore) Slice(channel Channel, fromMs, toMs int64) []byte {
	if a == nil {
		return nil
	}
	t, ok := a.tracks[channel]
	if !ok {
		return nil
	}
	off := t.offsetMs()
	from := fromMs - off
	if from < 0 {
		from = 0
	}
	to := int64(0)
	if toMs > 0 {
		to = toMs - off
		if to < 0 {
			to = 0
		}
	}
	return t.slice(from, to)
}

func (a *AudioStore) Close() {
	if a == nil {
		return
	}
	sidecar := filepath.Join(a.sessionDir, a.sessionID+".audio.json")
	doc := map[string]any{
		"session_id": a.sessionID,
		"started_at": a.startWall,
		"tracks": map[string]any{
			string(ChannelMicIn): map[string]any{
				"sample_rate": defaultMicSampleRate,
				"samples":     a.tracks[ChannelMicIn].totalSamples,
				"offset_ms":   a.tracks[ChannelMicIn].offsetMs(),
			},
			string(ChannelTTSOut): map[string]any{
				"sample_rate": defaultTTSSampleRate,
				"samples":     a.tracks[ChannelTTSOut].totalSamples,
				"offset_ms":   a.tracks[ChannelTTSOut].offsetMs(),
			},
		},
	}
	if data, err := json.Marshal(doc); err == nil {
		_ = os.WriteFile(sidecar, data, 0o644)
	}
	for _, t := range a.tracks {
		t.close()
	}
}

func WAVHeader(numSamples int, sampleRate int) []byte {
	const headerSize = 44
	out := make([]byte, headerSize)
	dataBytes := uint32(numSamples * 2)
	byteRate := uint32(sampleRate * 2)
	copy(out[0:4], []byte("RIFF"))
	binary.LittleEndian.PutUint32(out[4:8], 36+dataBytes)
	copy(out[8:12], []byte("WAVE"))
	copy(out[12:16], []byte("fmt "))
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1)
	binary.LittleEndian.PutUint16(out[22:24], 1)
	binary.LittleEndian.PutUint32(out[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:32], byteRate)
	binary.LittleEndian.PutUint16(out[32:34], 2)
	binary.LittleEndian.PutUint16(out[34:36], 16)
	copy(out[36:40], []byte("data"))
	binary.LittleEndian.PutUint32(out[40:44], dataBytes)
	return out
}
