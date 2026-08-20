package eou

import "strings"

type Turn struct {
	Role    string
	Content string
}

func FormatQwenChat(turns []Turn, partial string) string {
	var b strings.Builder
	for _, t := range turns {
		role := t.Role
		if role == "" {
			role = "user"
		}
		b.WriteString(ImStart)
		b.WriteString(role)
		b.WriteByte('\n')
		b.WriteString(t.Content)
		b.WriteString(ImEnd)
		b.WriteByte('\n')
	}
	if partial != "" {
		b.WriteString(ImStart)
		b.WriteString("user\n")
		b.WriteString(partial)
	}
	return b.String()
}

func RollingHistory(turns []Turn, maxTurns int) []Turn {
	if maxTurns <= 0 || len(turns) <= maxTurns {
		return turns
	}
	return turns[len(turns)-maxTurns:]
}
