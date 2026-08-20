package diarization

import "math"

const (
	defaultEMASmoothing float32 = 0.9
	maxEMASmoothing     float32 = 0.999
	normFloor           float32 = 1e-9
)

type ClusterID uint32

type centroid struct {
	id  ClusterID
	vec []float32
}

type OnlineClusterer struct {
	centroids    []centroid
	nextID       ClusterID
	threshold    float32
	maxSpeakers  int
	emaSmoothing float32
}

func NewOnlineClusterer(threshold float32, maxSpeakers int) *OnlineClusterer {
	if maxSpeakers < 1 {
		maxSpeakers = 1
	}
	return &OnlineClusterer{
		centroids:    make([]centroid, 0, maxSpeakers),
		threshold:    threshold,
		maxSpeakers:  maxSpeakers,
		emaSmoothing: defaultEMASmoothing,
	}
}

func (c *OnlineClusterer) WithEMA(s float32) *OnlineClusterer {
	switch {
	case s < 0:
		s = 0
	case s > maxEMASmoothing:
		s = maxEMASmoothing
	}
	c.emaSmoothing = s
	return c
}

func (c *OnlineClusterer) Reset() {
	c.centroids = c.centroids[:0]
	c.nextID = 0
}

func (c *OnlineClusterer) NumClusters() int { return len(c.centroids) }

func (c *OnlineClusterer) Assign(emb []float32) (ClusterID, float32) {
	idx, sim, ok := c.bestMatch(emb)
	if ok && sim >= c.threshold {
		c.updateCentroid(idx, emb)
		return c.centroids[idx].id, sim
	}
	if len(c.centroids) < c.maxSpeakers {
		id := c.nextID
		c.nextID++
		v := make([]float32, len(emb))
		copy(v, emb)
		c.centroids = append(c.centroids, centroid{id: id, vec: v})
		if ok {
			return id, sim
		}
		return id, 1.0
	}
	c.updateCentroid(idx, emb)
	return c.centroids[idx].id, sim
}

func (c *OnlineClusterer) Lookup(emb []float32) (ClusterID, float32, bool) {
	idx, sim, ok := c.bestMatch(emb)
	if !ok || sim < c.threshold {
		return 0, 0, false
	}
	return c.centroids[idx].id, sim, true
}

func (c *OnlineClusterer) bestMatch(emb []float32) (int, float32, bool) {
	bestIdx := -1
	bestSim := float32(math.Inf(-1))
	for i := range c.centroids {
		if len(c.centroids[i].vec) != len(emb) {
			continue
		}
		s := CosineSim(c.centroids[i].vec, emb)
		if bestIdx < 0 || s > bestSim {
			bestIdx = i
			bestSim = s
		}
	}
	if bestIdx < 0 {
		return 0, 0, false
	}
	return bestIdx, bestSim, true
}

func (c *OnlineClusterer) updateCentroid(idx int, emb []float32) {
	cv := c.centroids[idx].vec
	a := c.emaSmoothing
	var sum float32
	for i := range cv {
		cv[i] = a*cv[i] + (1-a)*emb[i]
		sum += cv[i] * cv[i]
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm < normFloor {
		norm = normFloor
	}
	inv := 1 / norm
	for i := range cv {
		cv[i] *= inv
	}
}

func CosineSim(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
