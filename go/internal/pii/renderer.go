package pii

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math/rand"
	"strconv"
	"strings"
)

type RenderRect struct {
	Left   int `json:"left"`
	Top    int `json:"top"`
	Right  int `json:"right"`
	Bottom int `json:"bottom"`
}

func DecodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

func RenderRedactions(img image.Image, rects []RenderRect, fillMode string, fillColor color.Color) ([]byte, error) {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Src)

	for _, r := range rects {
		rect := image.Rect(r.Left, r.Top, r.Right, r.Bottom).Intersect(bounds)
		if rect.Empty() {
			continue
		}
		switch fillMode {
		case "shuffle":
			fillShuffle(dst, rect)
		default:
			fillSolid(dst, rect, fillColor)
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, dst); err != nil {
		return nil, fmt.Errorf("encode png: %w", err)
	}
	return buf.Bytes(), nil
}

func fillSolid(dst *image.RGBA, rect image.Rectangle, c color.Color) {
	uniform := image.NewUniform(c)
	draw.Draw(dst, rect, uniform, image.Point{}, draw.Src)
}

func fillShuffle(dst *image.RGBA, rect image.Rectangle) {
	type bucket struct {
		r, g, b uint32
		count   int
	}

	hist := make(map[uint8]*bucket)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			quantKey := uint8((r>>12)<<4 | (g >> 12))
			if _, ok := hist[quantKey]; !ok {
				hist[quantKey] = &bucket{}
			}
			e := hist[quantKey]
			e.r += r >> 8
			e.g += g >> 8
			e.b += b >> 8
			e.count++
		}
	}

	var top1, top2 *bucket
	for _, b := range hist {
		if top1 == nil || b.count > top1.count {
			top2 = top1
			top1 = b
		} else if top2 == nil || b.count > top2.count {
			top2 = b
		}
	}

	if top1 == nil {
		return
	}
	if top2 == nil {
		top2 = top1
	}

	c1 := color.RGBA{
		R: uint8(top1.r / uint32(top1.count)),
		G: uint8(top1.g / uint32(top1.count)),
		B: uint8(top1.b / uint32(top1.count)),
		A: 255,
	}
	c2 := color.RGBA{
		R: uint8(top2.r / uint32(top2.count)),
		G: uint8(top2.g / uint32(top2.count)),
		B: uint8(top2.b / uint32(top2.count)),
		A: 255,
	}

	seed := int64(rect.Min.X ^ rect.Min.Y ^ rect.Max.X)
	rng := rand.New(rand.NewSource(seed))

	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			if rng.Intn(2) == 0 {
				dst.SetRGBA(x, y, c1)
			} else {
				dst.SetRGBA(x, y, c2)
			}
		}
	}
}

func ParseHexColor(s string) (color.Color, error) {
	s = strings.TrimPrefix(s, "#")
	if len(s) != 6 {
		return nil, fmt.Errorf("invalid hex color: %q", s)
	}
	r, err := strconv.ParseUint(s[0:2], 16, 8)
	if err != nil {
		return nil, fmt.Errorf("invalid hex color: %w", err)
	}
	g, err := strconv.ParseUint(s[2:4], 16, 8)
	if err != nil {
		return nil, fmt.Errorf("invalid hex color: %w", err)
	}
	b, err := strconv.ParseUint(s[4:6], 16, 8)
	if err != nil {
		return nil, fmt.Errorf("invalid hex color: %w", err)
	}
	return color.RGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255}, nil
}
