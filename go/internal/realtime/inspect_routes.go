package realtime

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/go-chi/chi/v5"

	"github.com/eordano/speaches-plus-go/internal/inspect"
	"github.com/eordano/speaches-plus-go/internal/oapi"
)

func (s *Server) sessionDir() string {
	if s != nil && s.cfg.InspectSessionDir != "" {
		return s.cfg.InspectSessionDir
	}
	return inspect.DefaultSessionDir()
}

func validSID(sid string) bool {
	if sid == "" || len(sid) > 64 {
		return false
	}
	for i := 0; i < len(sid); i++ {
		c := sid[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		return false
	}
	return true
}

func (s *Server) HandleInspectListSessions(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(inspect.ListMeta())
}

func (s *Server) HandleInspectListHistory(w http.ResponseWriter, _ *http.Request) {
	dir := s.sessionDir()
	entries, err := os.ReadDir(dir)
	if err != nil && !os.IsNotExist(err) {
		oapi.WriteError(w, http.StatusInternalServerError, err.Error(), oapi.TypeServerError, "", "")
		return
	}
	out := []inspect.SessionHistoryEntry{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if filepath.Ext(name) != ".ndjson" {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			continue
		}
		sid := strings.TrimSuffix(name, ".ndjson")
		out = append(out, inspect.SessionHistoryEntry{
			ID:        sid,
			SizeBytes: fi.Size(),
			MTime:     float64(fi.ModTime().UnixNano()) / 1e9,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].MTime > out[j].MTime })
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (s *Server) HandleInspectGetHistory(w http.ResponseWriter, r *http.Request) {
	sid := chi.URLParam(r, "sid")
	if !validSID(sid) {
		oapi.WriteError(w, http.StatusBadRequest, "invalid sid", oapi.TypeInvalidRequest, "sid", "invalid")
		return
	}
	path := filepath.Join(s.sessionDir(), sid+".ndjson")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			oapi.WriteError(w, http.StatusNotFound, "session not found", oapi.TypeNotFound, "sid", "session_not_found")
			return
		}
		oapi.WriteError(w, http.StatusInternalServerError, err.Error(), oapi.TypeServerError, "", "")
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/x-ndjson")
	_, _ = io.Copy(w, f)
}

func (s *Server) HandleInspectGetAudio(w http.ResponseWriter, r *http.Request) {
	sid := chi.URLParam(r, "sid")
	if !validSID(sid) {
		oapi.WriteError(w, http.StatusBadRequest, "invalid sid", oapi.TypeInvalidRequest, "sid", "invalid")
		return
	}
	channel := r.URL.Query().Get("channel")
	if channel != string(inspect.ChannelMicIn) && channel != string(inspect.ChannelTTSOut) {
		oapi.WriteError(w, http.StatusBadRequest, "invalid channel", oapi.TypeInvalidRequest, "channel", "invalid_value")
		return
	}
	fromMs, _ := strconv.ParseInt(r.URL.Query().Get("from_ms"), 10, 64)
	toMs, _ := strconv.ParseInt(r.URL.Query().Get("to_ms"), 10, 64)

	sr := inspect.ChannelSampleRate(inspect.Channel(channel))
	var pcm []byte

	if state := inspect.GetState(sid); state != nil {
		if as := getSessionAudioStore(sid); as != nil {
			pcm = as.Slice(inspect.Channel(channel), fromMs, toMs)
		}
	}
	if pcm == nil {
		raw := filepath.Join(s.sessionDir(), sid+".audio_"+channel+".raw")
		data, err := os.ReadFile(raw)
		if err != nil {
			if os.IsNotExist(err) {
				oapi.WriteError(w, http.StatusNotFound, "no audio for session", oapi.TypeNotFound, "sid", "audio_not_found")
				return
			}
			oapi.WriteError(w, http.StatusInternalServerError, err.Error(), oapi.TypeServerError, "", "")
			return
		}
		offsetMs := int64(0)
		sidecar := filepath.Join(s.sessionDir(), sid+".audio.json")
		if sb, err := os.ReadFile(sidecar); err == nil {
			var meta map[string]any
			if json.Unmarshal(sb, &meta) == nil {
				if tracks, ok := meta["tracks"].(map[string]any); ok {
					if c, ok := tracks[channel].(map[string]any); ok {
						if v, ok := c["offset_ms"].(float64); ok {
							offsetMs = int64(v)
						}
					}
				}
			}
		}
		from := (fromMs - offsetMs) * int64(sr) * 2 / 1000
		if from < 0 {
			from = 0
		}
		if from > int64(len(data)) {
			from = int64(len(data))
		}
		to := int64(len(data))
		if toMs > 0 {
			to = (toMs - offsetMs) * int64(sr) * 2 / 1000
			if to < from {
				to = from
			}
			if to > int64(len(data)) {
				to = int64(len(data))
			}
		}
		pcm = data[from:to]
	}

	body := bytes.NewBuffer(inspect.WAVHeader(len(pcm)/2, sr))
	body.Write(pcm)
	w.Header().Set("Content-Type", "audio/wav")
	_, _ = w.Write(body.Bytes())
}

func (s *Server) HandleInspectStream(w http.ResponseWriter, r *http.Request) {
	sid := chi.URLParam(r, "sid")
	if !validSID(sid) {
		oapi.WriteError(w, http.StatusBadRequest, "invalid sid", oapi.TypeInvalidRequest, "sid", "invalid")
		return
	}
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		Subprotocols:    []string{"inspect"},
		OriginPatterns:  []string{"*"},
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		oapi.WriteError(w, http.StatusBadRequest, "websocket upgrade failed: "+err.Error(), oapi.TypeInvalidRequest, "", "websocket_upgrade_failed")
		return
	}
	defer conn.CloseNow()

	relay := inspect.GetRelay(sid)
	if relay == nil {
		path := filepath.Join(s.sessionDir(), sid+".ndjson")
		f, err := os.Open(path)
		if err != nil {
			conn.Close(websocket.StatusNormalClosure, "session not found")
			return
		}
		defer f.Close()
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 64*1024), 1<<20)
		for sc.Scan() {
			line := sc.Bytes()
			if len(line) == 0 {
				continue
			}
			wctx, wcancel := context.WithTimeout(ctx, time.Duration(inspectStreamWriteTimeoutSec)*time.Second)
			err := conn.Write(wctx, websocket.MessageText, append([]byte(nil), line...))
			wcancel()
			if err != nil {
				return
			}
		}
		conn.Close(websocket.StatusNormalClosure, "")
		return
	}

	sub := relay.Subscribe()
	defer sub.Close()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	go func() {
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				cancel()
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case line, ok := <-sub.Channel():
			if !ok {
				return
			}
			payload := bytes.TrimRight(line, "\n")
			wctx, wcancel := context.WithTimeout(ctx, time.Duration(inspectStreamWriteTimeoutSec)*time.Second)
			err := conn.Write(wctx, websocket.MessageText, payload)
			wcancel()
			if err != nil {
				return
			}
		}
	}
}
