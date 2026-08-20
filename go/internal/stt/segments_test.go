package stt

import (
	"reflect"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/diarization"
)

func TestParseTimestampToken(t *testing.T) {
	cases := []struct {
		in     string
		want   uint32
		wantOk bool
	}{
		{"<|0.00|>", 0, true},
		{"<|1.20|>", 1200, true},
		{"<|29.98|>", 29980, true},
		{"<|0.5|>", 500, true},
		{"<|0.500|>", 500, true},
		{"<|sot|>", 0, false},
		{"<|en|>", 0, false},
		{"hello", 0, false},
		{"<||>", 0, false},
		{"<|.5|>", 0, false},
		{"<|5.|>", 0, false},
	}
	for _, c := range cases {
		got, ok := parseTimestampToken(c.in)
		if ok != c.wantOk || (ok && got != c.want) {
			t.Errorf("parseTimestampToken(%q) = (%d, %v), want (%d, %v)", c.in, got, ok, c.want, c.wantOk)
		}
	}
}

func TestParseCT2SegmentsFromTokens_TwoSegments(t *testing.T) {

	blob := []byte("<|sot|>\n<|en|>\n<|0.00|>\n hel\nlo\n<|1.20|>\n<|1.20|>\n wor\nld\n<|2.50|>\n")
	got := parseCT2SegmentsFromTokens(blob)
	if len(got) != 2 {
		t.Fatalf("want 2 segments, got %d: %#v", len(got), got)
	}
	if got[0].TStartMs != 0 || got[0].TEndMs != 1200 {
		t.Errorf("seg0 timing: %#v", got[0])
	}
	if got[0].Text != "hello" {
		t.Errorf("seg0 text = %q, want %q", got[0].Text, "hello")
	}
	if got[1].TStartMs != 1200 || got[1].TEndMs != 2500 {
		t.Errorf("seg1 timing: %#v", got[1])
	}
	if got[1].Text != "world" {
		t.Errorf("seg1 text = %q, want %q", got[1].Text, "world")
	}
}

func TestParseCT2SegmentsFromTokens_Empty(t *testing.T) {
	if got := parseCT2SegmentsFromTokens(nil); got != nil {
		t.Errorf("nil blob: want nil, got %#v", got)
	}
	if got := parseCT2SegmentsFromTokens([]byte("")); got != nil {
		t.Errorf("empty blob: want nil, got %#v", got)
	}

	if got := parseCT2SegmentsFromTokens([]byte("<|sot|>\n<|en|>\n")); got != nil {
		t.Errorf("specials-only: want nil, got %#v", got)
	}
}

func TestParseWhisperSegmentBlob(t *testing.T) {
	blob := []byte("0\t1200\t-0.3\t0.04\thello\n1200\t2500\t-0.5\t0.06\tworld\n")
	segs, err := parseWhisperSegmentBlob(blob)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("want 2 segments, got %d", len(segs))
	}
	if segs[0].TStartMs != 0 || segs[0].TEndMs != 1200 || segs[0].Text != "hello" {
		t.Errorf("seg0 = %#v", segs[0])
	}
	if segs[0].AvgLogprob == nil || *segs[0].AvgLogprob != -0.3 {
		t.Errorf("seg0 avg_logprob = %v", segs[0].AvgLogprob)
	}
	if segs[1].NoSpeechProb == nil || *segs[1].NoSpeechProb != 0.06 {
		t.Errorf("seg1 nsp = %v", segs[1].NoSpeechProb)
	}
}

func TestParseWhisperSegmentBlob_Empty(t *testing.T) {
	got, err := parseWhisperSegmentBlob(nil)
	if err != nil || got != nil {
		t.Errorf("nil: got=%v err=%v", got, err)
	}
}

func TestParseWhisperSegmentBlob_BadFraming(t *testing.T) {
	if _, err := parseWhisperSegmentBlob([]byte("not\ttabs\tenough\n")); err == nil {
		t.Error("expected error on malformed line")
	}
}

func TestBuildDiarizedResponse_AssignsByMidpoint(t *testing.T) {
	stt := Result{
		Text: "hello world goodbye",
		Segments: []Segment{
			{TStartMs: 0, TEndMs: 1000, Text: "hello"},
			{TStartMs: 1000, TEndMs: 2000, Text: "world"},
			{TStartMs: 2000, TEndMs: 3000, Text: "goodbye"},
		},
	}
	diar := []diarization.Segment{
		{Speaker: 0, TStartMs: 0, TEndMs: 1500, Confidence: 0.9},
		{Speaker: 1, TStartMs: 1500, TEndMs: 3000, Confidence: 0.8},
	}
	resp := buildDiarizedResponse(stt, diar)
	if resp.Text != "hello world goodbye" {
		t.Errorf("text passthrough = %q", resp.Text)
	}
	if len(resp.Segments) != 2 {
		t.Fatalf("want 2 diar segments, got %d", len(resp.Segments))
	}

	if resp.Segments[0].Speaker == nil || *resp.Segments[0].Speaker != "SPEAKER_00" {
		t.Errorf("seg0 speaker = %v", resp.Segments[0].Speaker)
	}
	if resp.Segments[0].Text != "hello world" {
		t.Errorf("seg0 text = %q, want %q", resp.Segments[0].Text, "hello world")
	}
	if resp.Segments[1].Text != "goodbye" {
		t.Errorf("seg1 text = %q, want %q", resp.Segments[1].Text, "goodbye")
	}
	if resp.Segments[0].Type != "transcript.text.segment" {
		t.Errorf("seg0 type = %q", resp.Segments[0].Type)
	}
	if resp.Segments[0].ID != "seg_001" || resp.Segments[1].ID != "seg_002" {
		t.Errorf("ids = %q, %q (want seg_001, seg_002)", resp.Segments[0].ID, resp.Segments[1].ID)
	}
}

func TestBuildDiarizedResponse_NoDiarFallsBackToWhisperSegments(t *testing.T) {
	stt := Result{
		Text: "alone",
		Segments: []Segment{
			{TStartMs: 0, TEndMs: 1000, Text: "alone"},
		},
	}
	resp := buildDiarizedResponse(stt, nil)
	if len(resp.Segments) != 1 {
		t.Fatalf("want 1 segment fallback, got %d", len(resp.Segments))
	}
	if resp.Segments[0].Speaker != nil {
		t.Errorf("expected nil speaker, got %v", resp.Segments[0].Speaker)
	}
	if resp.Segments[0].Text != "alone" {
		t.Errorf("text = %q", resp.Segments[0].Text)
	}
}

func TestNearestDiarIdx(t *testing.T) {
	diar := []diarization.Segment{
		{TStartMs: 0, TEndMs: 1000},
		{TStartMs: 2000, TEndMs: 3000},
		{TStartMs: 5000, TEndMs: 6000},
	}
	cases := []struct {
		mid  uint64
		want int
	}{
		{500, 0},
		{1500, 0},
		{1700, 1},
		{4000, 1},
		{10000, 2},
	}
	for _, c := range cases {
		if got := nearestDiarIdx(diar, c.mid); got != c.want {
			t.Errorf("nearestDiarIdx(%d) = %d, want %d", c.mid, got, c.want)
		}
	}
}

func TestAggregateSegmentStats_DurationWeighted(t *testing.T) {
	lpA, lpB := float32(-1.0), float32(-2.0)
	nspA, nspB := float32(0.1), float32(0.5)
	segs := []Segment{
		{TStartMs: 0, TEndMs: 100, AvgLogprob: &lpA, NoSpeechProb: &nspA},
		{TStartMs: 100, TEndMs: 1100, AvgLogprob: &lpB, NoSpeechProb: &nspB},
	}
	gotLP, gotNSP := aggregateSegmentStats(segs)
	wantLP := (-1.0*100 + -2.0*1000) / 1100.0
	wantNSP := (0.1*100 + 0.5*1000) / 1100.0
	if gotLP == nil || !floatNear(*gotLP, float32(wantLP)) {
		t.Errorf("lp = %v, want ~%v", gotLP, wantLP)
	}
	if gotNSP == nil || !floatNear(*gotNSP, float32(wantNSP)) {
		t.Errorf("nsp = %v, want ~%v", gotNSP, wantNSP)
	}
}

func TestAggregateSegmentStats_AllNil(t *testing.T) {
	segs := []Segment{{TStartMs: 0, TEndMs: 100, Text: "x"}}
	lp, nsp := aggregateSegmentStats(segs)
	if lp != nil || nsp != nil {
		t.Errorf("expected nil, got %v %v", lp, nsp)
	}
}

func TestJoinSegmentText_SkipsEmpty(t *testing.T) {
	got := joinSegmentText([]Segment{
		{Text: "hello"},
		{Text: ""},
		{Text: "world"},
	})
	want := "hello world"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func floatNear(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-4
}

func TestDiarizedResponseShape(t *testing.T) {
	got := reflect.TypeOf(diarizedResponse{})
	wantFields := []string{"Text", "AvgLogprob", "NoSpeechProb", "Segments"}
	if got.NumField() != len(wantFields) {
		t.Fatalf("field count drifted: got %d, want %d", got.NumField(), len(wantFields))
	}
	for i, name := range wantFields {
		if got.Field(i).Name != name {
			t.Errorf("field %d: got %q, want %q", i, got.Field(i).Name, name)
		}
	}

	seg := reflect.TypeOf(diarizedSegment{})
	wantSegFields := []string{
		"Type", "ID", "Speaker", "Start", "End", "Duration",
		"Text", "AvgLogprob", "NoSpeechProb", "Confidence",
	}
	if seg.NumField() != len(wantSegFields) {
		t.Fatalf("segment field count drifted: got %d, want %d", seg.NumField(), len(wantSegFields))
	}
	for i, name := range wantSegFields {
		if seg.Field(i).Name != name {
			t.Errorf("segment field %d: got %q, want %q", i, seg.Field(i).Name, name)
		}
	}
}
