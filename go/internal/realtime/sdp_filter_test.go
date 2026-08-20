package realtime

import (
	"strings"
	"testing"
)

func TestNormalizeOfferUnifiesIceCredsAcrossBundle(t *testing.T) {
	input := "v=0\r\n" +
		"a=group:BUNDLE 0 1\r\n" +
		"m=audio 1 UDP/TLS/RTP/SAVPF 96\r\n" +
		"a=mid:0\r\n" +
		"a=ice-ufrag:AAA1\r\n" +
		"a=ice-pwd:pwd-aaaa\r\n" +
		"m=application 1 DTLS/SCTP 5000\r\n" +
		"a=mid:1\r\n" +
		"a=ice-ufrag:BBB2\r\n" +
		"a=ice-pwd:pwd-bbbb\r\n"
	out := normalizeOffer(input)
	var ufrags, pwds []string
	for _, l := range strings.Split(out, "\r\n") {
		switch {
		case strings.HasPrefix(l, "a=ice-ufrag:"):
			ufrags = append(ufrags, l)
		case strings.HasPrefix(l, "a=ice-pwd:"):
			pwds = append(pwds, l)
		}
	}
	wantU := []string{"a=ice-ufrag:AAA1", "a=ice-ufrag:AAA1"}
	wantP := []string{"a=ice-pwd:pwd-aaaa", "a=ice-pwd:pwd-aaaa"}
	if !equalStrings(ufrags, wantU) {
		t.Fatalf("ufrags=%v want %v", ufrags, wantU)
	}
	if !equalStrings(pwds, wantP) {
		t.Fatalf("pwds=%v want %v", pwds, wantP)
	}
}

func TestNormalizeOfferIsIdentityWhenAlreadyUnified(t *testing.T) {
	input := "v=0\r\n" +
		"a=group:BUNDLE 0\r\n" +
		"m=audio 1 UDP/TLS/RTP/SAVPF 96\r\n" +
		"a=ice-ufrag:OnlyOne\r\n" +
		"a=ice-pwd:OnlyPwd\r\n"
	if got := normalizeOffer(input); got != input {
		t.Fatalf("expected identity\n got: %q\nwant: %q", got, input)
	}
}

func TestNormalizeOfferPreservesLineTerminator(t *testing.T) {

	input := "v=0\r\n" +
		"a=ice-ufrag:AAA1\r\n" +
		"m=audio 1 UDP/TLS/RTP/SAVPF 96\r\n" +
		"a=ice-ufrag:BBB2\n" +
		"a=ice-pwd:pwd-aaaa\r\n" +
		"a=ice-pwd:pwd-bbbb\n"
	out := normalizeOffer(input)
	if !strings.Contains(out, "a=ice-ufrag:AAA1\n") {
		t.Fatalf("missing rewritten lone-LF ufrag in:\n%s", out)
	}
	if strings.Contains(out, "BBB2") || strings.Contains(out, "pwd-bbbb") {
		t.Fatalf("stale credentials survived:\n%s", out)
	}
}

func TestNormalizeOfferEmpty(t *testing.T) {
	if got := normalizeOffer(""); got != "" {
		t.Fatalf("empty in -> %q", got)
	}
}

func TestNormalizeOfferNoIceCreds(t *testing.T) {

	input := "v=0\r\nm=audio 1 UDP/TLS/RTP/SAVPF 96\r\n"
	if got := normalizeOffer(input); got != input {
		t.Fatalf("got %q want %q", got, input)
	}
}

func TestSplitInclusiveRoundTrip(t *testing.T) {
	cases := []string{"", "a", "a\n", "\n", "a\nb", "a\nb\n", "abc\n\nxyz\n"}
	for _, c := range cases {
		var got string
		for _, p := range splitInclusive(c, '\n') {
			got += p
		}
		if got != c {
			t.Fatalf("round-trip: in=%q got %q", c, got)
		}
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
