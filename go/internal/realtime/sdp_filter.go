package realtime

import "strings"

func normalizeOffer(sdp string) string {
	var canonicalUfrag, canonicalPwd string
	haveUfrag, havePwd := false, false

	var b strings.Builder
	b.Grow(len(sdp))

	for _, line := range splitInclusive(sdp, '\n') {
		trimmed := strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(trimmed, "a=ice-ufrag:"):
			rest := trimmed[len("a=ice-ufrag:"):]
			if !haveUfrag {
				canonicalUfrag = rest
				haveUfrag = true
				b.WriteString(line)
			} else {
				b.WriteString("a=ice-ufrag:")
				b.WriteString(canonicalUfrag)
				b.WriteString(lineTerminator(line))
			}
		case strings.HasPrefix(trimmed, "a=ice-pwd:"):
			rest := trimmed[len("a=ice-pwd:"):]
			if !havePwd {
				canonicalPwd = rest
				havePwd = true
				b.WriteString(line)
			} else {
				b.WriteString("a=ice-pwd:")
				b.WriteString(canonicalPwd)
				b.WriteString(lineTerminator(line))
			}
		default:
			b.WriteString(line)
		}
	}
	return b.String()
}

func splitInclusive(s string, sep byte) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == sep {
			out = append(out, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func lineTerminator(line string) string {
	if strings.HasSuffix(line, "\r\n") {
		return "\r\n"
	}
	if strings.HasSuffix(line, "\n") {
		return "\n"
	}
	return ""
}

func filterOpusOnly(sdp string) string {
	lines := strings.Split(sdp, "\r\n")

	type mSection struct {
		start, end int
		opusPT     string
	}
	var sections []mSection
	cur := -1
	for i, l := range lines {
		if strings.HasPrefix(l, "m=audio") {
			if cur >= 0 {
				sections[cur].end = i
			}
			sections = append(sections, mSection{start: i, end: len(lines)})
			cur = len(sections) - 1
			continue
		}
		if strings.HasPrefix(l, "m=") && cur >= 0 {
			sections[cur].end = i
			cur = -1
		}
		if cur >= 0 && strings.HasPrefix(l, "a=rtpmap:") {
			rest := strings.TrimPrefix(l, "a=rtpmap:")
			parts := strings.SplitN(rest, " ", 2)
			if len(parts) == 2 && strings.HasPrefix(strings.ToLower(parts[1]), "opus/") {
				sections[cur].opusPT = parts[0]
			}
		}
	}

	out := make([]string, 0, len(lines))
	for i, l := range lines {
		section := -1
		for si, s := range sections {
			if i >= s.start && i < s.end {
				section = si
				break
			}
		}
		if section == -1 {
			out = append(out, l)
			continue
		}
		opusPT := sections[section].opusPT
		if opusPT == "" {
			out = append(out, l)
			continue
		}
		if strings.HasPrefix(l, "m=audio") {
			fields := strings.Fields(l)
			if len(fields) >= 4 {
				out = append(out, strings.Join(append(fields[:3], opusPT), " "))
			} else {
				out = append(out, l)
			}
			continue
		}
		if isPayloadAttr(l) {
			pt := payloadTypeOf(l)
			if pt != "" && pt != opusPT {
				continue
			}
		}
		out = append(out, l)
	}
	return strings.Join(out, "\r\n")
}

func isPayloadAttr(line string) bool {
	return strings.HasPrefix(line, "a=rtpmap:") ||
		strings.HasPrefix(line, "a=fmtp:") ||
		strings.HasPrefix(line, "a=rtcp-fb:")
}

func extractOpusChannels(sdp string) uint16 {
	for _, line := range strings.Split(sdp, "\r\n") {
		if !strings.HasPrefix(line, "a=rtpmap:") {
			continue
		}
		rest := strings.TrimPrefix(line, "a=rtpmap:")
		parts := strings.SplitN(rest, " ", 2)
		if len(parts) != 2 {
			continue
		}
		spec := strings.ToLower(parts[1])
		if !strings.HasPrefix(spec, "opus/") {
			continue
		}

		segs := strings.Split(spec, "/")
		if len(segs) >= 3 {
			switch segs[2] {
			case "1":
				return 1
			case "2":
				return 2
			}
		}
		return 1
	}
	return 1
}

func payloadTypeOf(line string) string {
	idx := strings.Index(line, ":")
	if idx < 0 {
		return ""
	}
	rest := line[idx+1:]
	sp := strings.Index(rest, " ")
	if sp < 0 {
		return rest
	}
	return rest[:sp]
}
