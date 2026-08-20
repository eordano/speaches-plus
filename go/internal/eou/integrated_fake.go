package eou

type FakeIntegratedScript struct {
	Signals []IntegratedSignal
}

type fakeIntegrated struct {
	out chan IntegratedSignal
}

func NewFakeIntegrated(script FakeIntegratedScript) IntegratedSource {
	out := make(chan IntegratedSignal, len(script.Signals)+1)
	for _, s := range script.Signals {
		out <- s
	}
	close(out)
	return &fakeIntegrated{out: out}
}

func (f *fakeIntegrated) Signals() <-chan IntegratedSignal { return f.out }
