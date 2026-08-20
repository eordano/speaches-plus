package diarization

import (
	"math"
	"testing"
)

func unit(v []float32) []float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	n := float32(math.Sqrt(float64(s)))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

func TestFirstEmbeddingCreatesClusterZero(t *testing.T) {
	c := NewOnlineClusterer(0.5, 4)
	id, _ := c.Assign(unit([]float32{1, 0, 0}))
	if id != 0 {
		t.Fatalf("want id 0, got %d", id)
	}
	if c.NumClusters() != 1 {
		t.Fatalf("want 1 cluster, got %d", c.NumClusters())
	}
}

func TestSimilarEmbeddingJoinsCluster(t *testing.T) {
	c := NewOnlineClusterer(0.5, 4)
	id1, _ := c.Assign(unit([]float32{1, 0, 0}))
	id2, sim := c.Assign(unit([]float32{0.99, 0.01, 0}))
	if id1 != id2 {
		t.Fatalf("want same cluster, got %d vs %d", id1, id2)
	}
	if sim <= 0.9 {
		t.Fatalf("want sim > 0.9, got %f", sim)
	}
	if c.NumClusters() != 1 {
		t.Fatalf("want 1 cluster, got %d", c.NumClusters())
	}
}

func TestDissimilarEmbeddingCreatesNewCluster(t *testing.T) {
	c := NewOnlineClusterer(0.5, 4)
	id1, _ := c.Assign(unit([]float32{1, 0, 0}))
	id2, _ := c.Assign(unit([]float32{0, 1, 0}))
	if id1 == id2 {
		t.Fatalf("want different ids, got %d == %d", id1, id2)
	}
	if c.NumClusters() != 2 {
		t.Fatalf("want 2 clusters, got %d", c.NumClusters())
	}
}

func TestMaxSpeakersCapsClusterCreation(t *testing.T) {
	c := NewOnlineClusterer(0.99, 2)
	c.Assign(unit([]float32{1, 0, 0}))
	c.Assign(unit([]float32{0, 1, 0}))
	c.Assign(unit([]float32{0, 0, 1}))
	if c.NumClusters() != 2 {
		t.Fatalf("want 2 clusters at cap, got %d", c.NumClusters())
	}
}

func TestLookupDoesNotCreateClusters(t *testing.T) {
	c := NewOnlineClusterer(0.5, 4)
	c.Assign(unit([]float32{1, 0, 0}))
	if _, _, ok := c.Lookup(unit([]float32{0, 1, 0})); ok {
		t.Fatalf("Lookup returned a match below threshold")
	}
	if c.NumClusters() != 1 {
		t.Fatalf("want 1 cluster, got %d", c.NumClusters())
	}
}

func TestResetClearsState(t *testing.T) {
	c := NewOnlineClusterer(0.5, 4)
	c.Assign(unit([]float32{1, 0, 0}))
	c.Reset()
	if c.NumClusters() != 0 {
		t.Fatalf("want 0 clusters after reset, got %d", c.NumClusters())
	}
	id, _ := c.Assign(unit([]float32{1, 0, 0}))
	if id != 0 {
		t.Fatalf("nextID should restart at 0 after reset, got %d", id)
	}
}
