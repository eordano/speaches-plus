package realtime

type (
	SessionID  string
	ItemID     string
	ResponseID string
	EventID    string
	Epoch      uint64
	Millis     int64
	DurationMs int64
	Samples    int
)

type SampleRate int

func (s Samples) ToMillis(sr SampleRate) Millis {
	return Millis(int64(s) * 1000 / int64(sr))
}

func (m Millis) Add(d DurationMs) Millis {
	return Millis(int64(m) + int64(d))
}

func (m Millis) Sub(other Millis) DurationMs {
	return DurationMs(int64(m) - int64(other))
}
