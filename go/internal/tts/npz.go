package tts

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type Voice struct {
	Shape []int
	Data  []float32
}

func LoadVoicesNPZ(path string) (map[string]Voice, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open zip: %w", err)
	}
	defer zr.Close()

	out := make(map[string]Voice, len(zr.File))
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		if name == f.Name {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open %s: %w", f.Name, err)
		}
		body, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", f.Name, err)
		}
		v, err := parseNPY(body)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", f.Name, err)
		}
		out[name] = v
	}
	return out, nil
}

func parseNPY(b []byte) (Voice, error) {
	if len(b) < 10 || string(b[:6]) != npyMagic {
		return Voice{}, fmt.Errorf("not an .npy file")
	}
	major, minor := b[6], b[7]
	if major < 1 || major > 3 {
		return Voice{}, fmt.Errorf("unsupported npy version %d.%d", major, minor)
	}
	var headerLen int
	var dataOff int
	switch major {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(b[8:10]))
		dataOff = 10 + headerLen
	default:
		headerLen = int(binary.LittleEndian.Uint32(b[8:12]))
		dataOff = 12 + headerLen
	}
	if dataOff > len(b) {
		return Voice{}, fmt.Errorf("npy header truncated")
	}
	headerOff := dataOff - headerLen
	header := string(b[headerOff:dataOff])

	descr, fortran, shape, err := parseNPYHeader(header)
	if err != nil {
		return Voice{}, err
	}
	if fortran {
		return Voice{}, fmt.Errorf("fortran-order arrays are not supported")
	}
	if descr != "<f4" {
		return Voice{}, fmt.Errorf("unsupported dtype %q (need <f4)", descr)
	}

	total := 1
	for _, d := range shape {
		total *= d
	}
	if dataOff+total*4 > len(b) {
		return Voice{}, fmt.Errorf("data shorter than shape implies (%d < %d)", len(b)-dataOff, total*4)
	}

	out := make([]float32, total)
	rd := bytes.NewReader(b[dataOff : dataOff+total*4])
	if err := binary.Read(rd, binary.LittleEndian, out); err != nil {
		return Voice{}, fmt.Errorf("decode floats: %w", err)
	}
	return Voice{Shape: shape, Data: out}, nil
}

func parseNPYHeader(s string) (descr string, fortran bool, shape []int, err error) {
	s = strings.TrimSpace(s)
	descr = stringField(s, "'descr':")
	fortranStr := tokenAfter(s, "'fortran_order':")
	switch fortranStr {
	case "True":
		fortran = true
	case "False":
		fortran = false
	default:
		err = fmt.Errorf("malformed fortran_order: %q", fortranStr)
		return
	}
	shapeStr := tupleAfter(s, "'shape':")
	for _, p := range strings.Split(shapeStr, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, perr := strconv.Atoi(p)
		if perr != nil {
			err = fmt.Errorf("malformed shape entry %q: %w", p, perr)
			return
		}
		shape = append(shape, n)
	}
	if descr == "" || len(shape) == 0 {
		err = fmt.Errorf("missing descr or shape in header: %q", s)
	}
	return
}

func stringField(s, key string) string {
	i := strings.Index(s, key)
	if i < 0 {
		return ""
	}
	rest := s[i+len(key):]
	q := strings.IndexByte(rest, '\'')
	if q < 0 {
		return ""
	}
	end := strings.IndexByte(rest[q+1:], '\'')
	if end < 0 {
		return ""
	}
	return rest[q+1 : q+1+end]
}

func tokenAfter(s, key string) string {
	i := strings.Index(s, key)
	if i < 0 {
		return ""
	}
	rest := strings.TrimSpace(s[i+len(key):])
	end := strings.IndexAny(rest, ", }")
	if end < 0 {
		return rest
	}
	return rest[:end]
}

func tupleAfter(s, key string) string {
	i := strings.Index(s, key)
	if i < 0 {
		return ""
	}
	rest := s[i+len(key):]
	open := strings.IndexByte(rest, '(')
	close := strings.IndexByte(rest, ')')
	if open < 0 || close < 0 || close <= open {
		return ""
	}
	return rest[open+1 : close]
}
