package inspect

import (
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func CleanupOnStartup(sessionDir string, maxCount int, maxBytes int64, maxAgeDays int) {
	if sessionDir == "" {
		return
	}
	entries, err := os.ReadDir(sessionDir)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("inspect.retention: read dir failed", "err", err, "path", sessionDir)
		}
		return
	}
	type sess struct {
		id    string
		paths []string
		mtime time.Time
		bytes int64
	}
	bySession := map[string]*sess{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		ext := filepath.Ext(name)
		if ext != ".ndjson" && ext != ".raw" && ext != ".json" {
			continue
		}
		sid := name
		if i := strings.IndexByte(sid, '.'); i >= 0 {
			sid = sid[:i]
		}
		path := filepath.Join(sessionDir, name)
		fi, err := e.Info()
		if err != nil {
			continue
		}
		s, ok := bySession[sid]
		if !ok {
			s = &sess{id: sid}
			bySession[sid] = s
		}
		s.paths = append(s.paths, path)
		s.bytes += fi.Size()
		if fi.ModTime().After(s.mtime) {
			s.mtime = fi.ModTime()
		}
	}

	now := time.Now()
	if maxAgeDays > 0 {
		cutoff := now.Add(-time.Duration(maxAgeDays) * 24 * time.Hour)
		for sid, s := range bySession {
			if s.mtime.Before(cutoff) {
				deleteSession(s.paths)
				delete(bySession, sid)
			}
		}
	}

	ordered := make([]*sess, 0, len(bySession))
	for _, s := range bySession {
		ordered = append(ordered, s)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].mtime.After(ordered[j].mtime) })

	if maxCount > 0 && len(ordered) > maxCount {
		for _, s := range ordered[maxCount:] {
			deleteSession(s.paths)
			delete(bySession, s.id)
		}
		ordered = ordered[:maxCount]
	}

	if maxBytes > 0 {
		var running int64
		for _, s := range ordered {
			if running+s.bytes > maxBytes {
				deleteSession(s.paths)
				delete(bySession, s.id)
			} else {
				running += s.bytes
			}
		}
	}
}

func deleteSession(paths []string) {
	for _, p := range paths {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			slog.Warn("inspect.retention: remove failed", "err", err, "path", p)
		}
	}
}
