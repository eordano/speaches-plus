package inspect

const (
	defaultRelayCap      = 1024
	defaultMicSampleRate = 16000
	defaultTTSSampleRate = 24000

	wsWriteTimeoutSec = 5

	defaultMaxSessions = 200
	defaultMaxBytes    = int64(2) << 30
	defaultMaxAgeDays  = 14
	historyChunkSize   = 65536

	defaultSessionDirEnv = "SPEACHES_INSPECT_SESSION_DIR"
	defaultSessionDirRel = ".cache/speaches/sessions"
)

const (
	LaneAudioLevel  LaneID = "audio_level"
	LaneVAD         LaneID = "vad"
	LaneSTT         LaneID = "stt"
	LaneTurn        LaneID = "turn"
	LaneBargein     LaneID = "bargein"
	LaneEOU         LaneID = "eou"
	LaneDiarization LaneID = "diarization"
	LaneLLM         LaneID = "llm"
	LaneResponse    LaneID = "response"
	LaneTool        LaneID = "tool"
	LaneTTSReq      LaneID = "tts_req"
	LaneTTSChunk    LaneID = "tts_chunk"
	LaneTTSPacer    LaneID = "tts_pacer"
	LaneWire        LaneID = "wire"
	LaneState       LaneID = "state"
	LaneError       LaneID = "error"
)

const (
	ChannelMicIn  Channel = "mic_in"
	ChannelTTSOut Channel = "tts_out"
)
