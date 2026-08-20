package audio

import (
	"math"
	"testing"
)

func TestG711_ULawRoundTripBytes(t *testing.T) {
	for code := 0; code < 256; code++ {
		if code == 0x7F {
			continue
		}
		linear := ulawDecodeTable[code]
		again := linearToUlaw(linear)
		if again != byte(code) {
			t.Errorf("µ-law roundtrip mismatch: code=0x%02x linear=%d -> 0x%02x", code, linear, again)
		}
	}
}

func TestG711_ALawRoundTripBytes(t *testing.T) {
	for code := 0; code < 256; code++ {
		linear := alawDecodeTable[code]
		again := linearToAlaw(linear)
		if again != byte(code) {
			t.Errorf("A-law roundtrip mismatch: code=0x%02x linear=%d -> 0x%02x", code, linear, again)
		}
	}
}

func TestG711_ULawSilence(t *testing.T) {
	silenceCode := linearToUlaw(0)
	if silenceCode != 0xFF {
		t.Errorf("µ-law silence: want 0xFF, got 0x%02x", silenceCode)
	}
	if d := ulawDecodeTable[0xFF]; d != 0 {
		t.Errorf("µ-law decode 0xFF: want 0, got %d", d)
	}
}

func TestG711_ALawSilence(t *testing.T) {
	silenceCode := linearToAlaw(0)
	if silenceCode != 0xD5 {
		t.Errorf("A-law silence: want 0xD5, got 0x%02x", silenceCode)
	}
}

func TestG711_F32ULawRoundTrip(t *testing.T) {
	in := MonoF32{0, 0.25, -0.25, 0.5, -0.5, 0.99, -0.99}
	bytes := F32ToULawBytes(in)
	back := ULawBytesToF32(bytes)
	if len(back) != len(in) {
		t.Fatalf("length mismatch: in=%d back=%d", len(in), len(back))
	}
	for i, want := range in {
		got := back[i]
		if math.Abs(float64(got-want)) > 0.05 {
			t.Errorf("sample %d: want %.4f got %.4f (diff %.4f)", i, want, got, got-want)
		}
	}
}

func TestG711_F32ALawRoundTrip(t *testing.T) {
	in := MonoF32{0, 0.25, -0.25, 0.5, -0.5, 0.99, -0.99}
	bytes := F32ToALawBytes(in)
	back := ALawBytesToF32(bytes)
	if len(back) != len(in) {
		t.Fatalf("length mismatch: in=%d back=%d", len(in), len(back))
	}
	for i, want := range in {
		got := back[i]
		if math.Abs(float64(got-want)) > 0.05 {
			t.Errorf("sample %d: want %.4f got %.4f (diff %.4f)", i, want, got, got-want)
		}
	}
}
