package eou

var byteToRune = func() [256]rune {
	var m [256]rune
	keep := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	bs := []int{}
	for b := 0; b < 256; b++ {
		if keep(b) {
			bs = append(bs, b)
		}
	}
	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		if !keep(b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i, b := range bs {
		m[b] = rune(cs[i])
	}
	return m
}()

var runeToByte = func() map[rune]byte {
	m := make(map[rune]byte, 256)
	for b, r := range byteToRune {
		m[r] = byte(b)
	}
	return m
}()

func bytesToBPETokens(s string) string {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, byteToRune[s[i]])
	}
	return string(out)
}

func bpeTokensToString(s string) string {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := runeToByte[r]; ok {
			out = append(out, b)
		}
	}
	return string(out)
}
