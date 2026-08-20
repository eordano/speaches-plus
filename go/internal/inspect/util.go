package inspect

import (
	"math"
	"os"
	"path/filepath"
)

func float64Bits(v float64) uint64     { return math.Float64bits(v) }
func float64FromBits(u uint64) float64 { return math.Float64frombits(u) }

func DefaultSessionDir() string {
	if v := os.Getenv(defaultSessionDirEnv); v != "" {
		return os.ExpandEnv(v)
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return defaultSessionDirRel
	}
	return filepath.Join(home, defaultSessionDirRel)
}
