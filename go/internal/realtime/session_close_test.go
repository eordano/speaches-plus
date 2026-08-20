package realtime

import (
	"sync"
	"testing"
)

func TestSessionPipelineCloseIsConcurrencySafe(t *testing.T) {
	const goroutines = 64
	for trial := 0; trial < 200; trial++ {
		p := &sessionPipeline{
			closed: make(chan struct{}),
		}
		var wg sync.WaitGroup
		start := make(chan struct{})
		for i := 0; i < goroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				p.close()
			}()
		}
		close(start)
		wg.Wait()

		select {
		case <-p.closed:
		default:
			t.Fatalf("trial %d: p.closed was not closed after p.close()", trial)
		}
	}
}
