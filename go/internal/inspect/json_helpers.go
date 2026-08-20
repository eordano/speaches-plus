package inspect

import (
	"encoding/json"
	"time"
)

func jsonMarshalLine(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func nowMonoNS() int64 { return time.Now().UnixNano() }
func nowWall() float64 { return float64(time.Now().UnixNano()) / 1e9 }
