package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/coder/websocket"

	"github.com/eordano/speaches-plus-go/internal/oapi"
)

type wsTransport interface {
	Write(ctx context.Context, typ websocket.MessageType, data []byte) error
}

func defaultLogger() *slog.Logger { return slog.Default() }

func jsonMarshalEvent(event any) ([]byte, error) { return json.Marshal(event) }

func (s *Server) HandleRealtimeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		Subprotocols:    []string{"realtime"},
		OriginPatterns:  []string{"*"},
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		oapi.WriteError(w, http.StatusBadRequest,
			"websocket upgrade failed: "+err.Error(),
			oapi.TypeInvalidRequest, "", "websocket_upgrade_failed")
		return
	}
	defer conn.CloseNow()

	conn.SetReadLimit(8 * 1024 * 1024)

	model := r.URL.Query().Get("model")
	if model == "" {
		conn.Close(websocket.StatusPolicyViolation, "missing model")
		return
	}
	cfg := s.makeSessionConfig(
		model,
		firstNonEmpty(r.URL.Query().Get("intent"), "conversation"),
		r.URL.Query().Get("transcription_model"),
		r.URL.Query().Get("voice"),
		r.URL.Query().Get("speech_model"),
		r.URL.Query().Get("language"),
	)
	cfg.Conversation = (cfg.Intent == "conversation")

	logger := defaultLogger()
	vadAdp, vadErr := newVADAdapter(s.cfg.SileroVADPath)
	if vadErr != nil {
		logger.Warn("silero unavailable for WS session, falling back to silence-timeout",
			"err", vadErr, "path", s.cfg.SileroVADPath)
		vadAdp = nil
	}

	pipeline := newSessionPipelineWithID(s.cfg, cfg, logger, newSessID())
	pipeline.vad = vadAdp
	pipeline.attachWSConn(conn)
	defer pipeline.close()

	if vadAdp == nil {

		pipeline.wg.Add(1)
		go func() {
			defer pipeline.wg.Done()
			pipeline.silenceWatchdog()
		}()
	}

	if cfg.Conversation {
		out := newWSOutbound(conn, pipeline.logger, cfg.OutputAudioFormat, newEventID, &pipeline.wsWriteMu)
		pipeline.attachOutbound(out)
	}

	pipeline.emitWSSessionCreated()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				pipeline.logger.Info("websocket read ended", "err", err)
			}
			return
		}
		if typ != websocket.MessageText {
			continue
		}
		pipeline.handleClientEvent(data)
	}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func (p *sessionPipeline) attachWSConn(conn *websocket.Conn) {
	p.chMu.Lock()
	if p.wsConn != nil {
		p.chMu.Unlock()
		return
	}
	p.wsConn = conn
	close(p.chReady)
	p.chMu.Unlock()
}

func (p *sessionPipeline) emitWSSessionCreated() {
	if p.wsConn == nil {
		return
	}
	ev := sessionCreatedEvent{
		EventID: newEventID(),
		Type:    SETSessionCreated,
		Session: session{
			ID:                p.sessionID,
			Object:            "realtime.session",
			Model:             p.session.Model,
			Modalities:        []string{"text", "audio"},
			InputAudioFormat:  p.session.InputAudioFormat,
			OutputAudioFormat: p.session.OutputAudioFormat,
		},
	}
	p.sendWS(ev, ev.EventID)
}

func (p *sessionPipeline) sendWS(event any, eventID string) {
	if p.wsConn == nil {
		return
	}
	body, err := jsonMarshalEvent(event)
	if err != nil {
		p.logger.Error("ws marshal failed", "err", err, "id", eventID)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), wsWriteTimeoutSec*time.Second)
	defer cancel()
	p.wsWriteMu.Lock()
	defer p.wsWriteMu.Unlock()
	if err := p.wsConn.Write(ctx, websocket.MessageText, body); err != nil {
		p.logger.Warn("ws write failed", "err", err, "id", eventID)
	}
}
