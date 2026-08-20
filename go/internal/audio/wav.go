package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

type WAV struct {
	Samples    MonoF32
	SampleRate int
}

func DecodeWAV(r io.Reader) (*WAV, error) {
	all, err := io.ReadAll(io.LimitReader(r, 200<<20))
	if err != nil {
		return nil, err
	}
	if len(all) < 44 {
		return nil, fmt.Errorf("wav: too short (%d bytes)", len(all))
	}
	br := bytes.NewReader(all)
	var hdr struct {
		ChunkID   [4]byte
		ChunkSize uint32
		Format    [4]byte
	}
	if err := binary.Read(br, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}
	if string(hdr.ChunkID[:]) != "RIFF" || string(hdr.Format[:]) != "WAVE" {
		return nil, fmt.Errorf("wav: not a RIFF/WAVE file")
	}

	var (
		sampleRate    uint32
		numChannels   uint16
		bitsPerSample uint16
		dataChunk     []byte
		formatTag     uint16
	)

	for br.Len() >= 8 {
		var ch struct {
			ID   [4]byte
			Size uint32
		}
		if err := binary.Read(br, binary.LittleEndian, &ch); err != nil {
			return nil, err
		}
		size := int(ch.Size)
		if size < 0 || size > br.Len() {
			size = br.Len()
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(br, body); err != nil {
			return nil, err
		}
		if size%2 == 1 {
			_, _ = br.ReadByte()
		}
		switch string(ch.ID[:]) {
		case "fmt ":
			if len(body) < 16 {
				return nil, fmt.Errorf("wav: fmt chunk too small (%d)", len(body))
			}
			formatTag = binary.LittleEndian.Uint16(body[0:2])
			numChannels = binary.LittleEndian.Uint16(body[2:4])
			sampleRate = binary.LittleEndian.Uint32(body[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
		case "data":
			dataChunk = body
		}
	}

	if dataChunk == nil {
		return nil, fmt.Errorf("wav: missing data chunk")
	}
	if formatTag != 1 {
		return nil, fmt.Errorf("wav: unsupported format tag %d (need PCM)", formatTag)
	}
	if numChannels == 0 {
		return nil, fmt.Errorf("wav: zero channels")
	}

	bytesPerSample := int(bitsPerSample / 8)
	frameSize := bytesPerSample * int(numChannels)
	if frameSize == 0 {
		return nil, fmt.Errorf("wav: zero frame size")
	}
	nFrames := len(dataChunk) / frameSize

	samples := make(MonoF32, nFrames)
	switch bitsPerSample {
	case 16:
		scale := float32(1.0 / 32768.0)
		for i := 0; i < nFrames; i++ {
			var sum int32
			for c := 0; c < int(numChannels); c++ {
				off := (i*int(numChannels) + c) * 2
				sum += int32(int16(binary.LittleEndian.Uint16(dataChunk[off : off+2])))
			}
			samples[i] = float32(sum) / float32(numChannels) * scale
		}
	case 24:
		scale := float32(1.0 / 8388608.0)
		for i := 0; i < nFrames; i++ {
			var sum int64
			for c := 0; c < int(numChannels); c++ {
				off := (i*int(numChannels) + c) * 3
				v := int32(dataChunk[off]) |
					int32(dataChunk[off+1])<<8 |
					int32(dataChunk[off+2])<<16
				if v&0x800000 != 0 {
					v |= ^0xFFFFFF
				}
				sum += int64(v)
			}
			samples[i] = float32(sum) / float32(numChannels) * scale
		}
	case 32:
		scale := float32(1.0 / 2147483648.0)
		for i := 0; i < nFrames; i++ {
			var sum int64
			for c := 0; c < int(numChannels); c++ {
				off := (i*int(numChannels) + c) * 4
				v := int32(binary.LittleEndian.Uint32(dataChunk[off : off+4]))
				sum += int64(v)
			}
			samples[i] = float32(sum) / float32(numChannels) * scale
		}
	default:
		return nil, fmt.Errorf("wav: unsupported bit depth %d", bitsPerSample)
	}

	return &WAV{Samples: samples, SampleRate: int(sampleRate)}, nil
}

func EncodeWAVMono16(samples MonoF32, sampleRate int) []byte {
	const numChannels uint16 = 1
	const bitsPerSample uint16 = 16
	const bytesPerSample = int(bitsPerSample / 8)
	dataSize := len(samples) * bytesPerSample * int(numChannels)
	totalSize := 36 + dataSize
	buf := bytes.NewBuffer(make([]byte, 0, 44+dataSize))
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(totalSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, numChannels)
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate*int(numChannels)*bytesPerSample))
	_ = binary.Write(buf, binary.LittleEndian, uint16(int(numChannels)*bytesPerSample))
	_ = binary.Write(buf, binary.LittleEndian, bitsPerSample)
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	for _, s := range samples {
		v := int32(s * 32767.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		_ = binary.Write(buf, binary.LittleEndian, int16(v))
	}
	return buf.Bytes()
}
