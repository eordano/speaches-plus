package pii

import "math"

const negInf = -1e30

type tag struct {
	prefix string
	cls    string
}

func splitLabel(label string) tag {
	if label == "O" {
		return tag{"O", ""}
	}
	for i := 0; i < len(label); i++ {
		if label[i] == '-' {
			p := label[:i]
			if p != "B" && p != "I" && p != "E" && p != "S" {
				return tag{"?", ""}
			}
			return tag{p, label[i+1:]}
		}
	}
	return tag{"?", ""}
}

func transitionAllowed(a, b tag) bool {
	switch a.prefix {
	case "O", "E", "S":
		return b.prefix == "O" || b.prefix == "B" || b.prefix == "S"
	case "B", "I":
		return (b.prefix == "I" || b.prefix == "E") && b.cls == a.cls
	}
	return false
}

func buildStart(labels []string) []float64 {
	out := make([]float64, len(labels))
	for i, l := range labels {
		p := splitLabel(l).prefix
		if p == "I" || p == "E" {
			out[i] = negInf
		}
	}
	return out
}

func buildTransitions(labels []string) [][]float64 {
	tags := make([]tag, len(labels))
	for i, l := range labels {
		tags[i] = splitLabel(l)
	}
	n := len(labels)
	out := make([][]float64, n)
	for i := range out {
		row := make([]float64, n)
		for j := range row {
			if transitionAllowed(tags[i], tags[j]) {
				row[j] = 0.0
			} else {
				row[j] = negInf
			}
		}
		out[i] = row
	}
	return out
}

func logSoftmax(row []float32) []float64 {
	m := float64(row[0])
	for _, v := range row[1:] {
		if float64(v) > m {
			m = float64(v)
		}
	}
	out := make([]float64, len(row))
	var sum float64
	for i, v := range row {
		out[i] = float64(v) - m
		sum += math.Exp(out[i])
	}
	logSum := math.Log(sum)
	for i := range out {
		out[i] -= logSum
	}
	return out
}

func ViterbiDecode(logits [][]float32, labels []string) []int32 {
	T := len(logits)
	if T == 0 {
		return nil
	}
	L := len(labels)

	trans := buildTransitions(labels)
	start := buildStart(labels)

	lp0 := logSoftmax(logits[0])
	delta := make([]float64, L)
	for j := 0; j < L; j++ {
		delta[j] = start[j] + lp0[j]
	}

	bp := make([][]int32, T)
	bp[0] = make([]int32, L)

	for t := 1; t < T; t++ {
		lp := logSoftmax(logits[t])
		newDelta := make([]float64, L)
		bpRow := make([]int32, L)
		for j := 0; j < L; j++ {
			bestScore := math.Inf(-1)
			bestPrev := int32(0)
			for i := 0; i < L; i++ {
				s := delta[i] + trans[i][j]
				if s > bestScore {
					bestScore = s
					bestPrev = int32(i)
				}
			}
			newDelta[j] = bestScore + lp[j]
			bpRow[j] = bestPrev
		}
		delta = newDelta
		bp[t] = bpRow
	}

	out := make([]int32, T)
	best := int32(0)
	bestVal := delta[0]
	for j := 1; j < L; j++ {
		if delta[j] > bestVal {
			bestVal = delta[j]
			best = int32(j)
		}
	}
	out[T-1] = best
	for t := T - 1; t > 0; t-- {
		out[t-1] = bp[t][out[t]]
	}
	return out
}
