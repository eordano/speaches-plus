package stt

import "unicode/utf8"

var bpeRuneToByte = func() map[rune]byte {
	m := make(map[rune]byte, 256)
	bs := []int{}
	for c := '!'; c <= '~'; c++ {
		bs = append(bs, int(c))
	}
	for c := '¡'; c <= '¬'; c++ {
		bs = append(bs, int(c))
	}
	for c := '®'; c <= 'ÿ'; c++ {
		bs = append(bs, int(c))
	}
	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i, codepoint := range cs {
		m[rune(codepoint)] = byte(bs[i])
	}
	return m
}()

func DecodeBPE(s string) string {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := bpeRuneToByte[r]; ok {
			out = append(out, b)
		} else {
			var buf [4]byte
			n := utf8.EncodeRune(buf[:], r)
			out = append(out, buf[:n]...)
		}
	}
	return string(out)
}
