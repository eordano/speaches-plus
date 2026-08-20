package punkt

import "testing"

func TestEnglishTrainedLoads(t *testing.T) {
	p, err := EnglishTrained()
	if err != nil {
		t.Fatal(err)
	}
	if len(p.OrthoContext) < 10000 || len(p.SentStarters) < 10 || len(p.Collocations) == 0 {
		t.Fatalf("suspiciously small params: ortho=%d starters=%d collocs=%d",
			len(p.OrthoContext), len(p.SentStarters), len(p.Collocations))
	}
}

func TestMultilingualLoads(t *testing.T) {
	for _, lang := range []string{"german", "spanish", "portuguese", "french"} {
		p, err := Trained(lang)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.OrthoContext) < 1000 || len(p.AbbrevTypes) == 0 {
			t.Fatalf("%s: ortho=%d abbrevs=%d", lang, len(p.OrthoContext), len(p.AbbrevTypes))
		}
	}
	ru, err := Trained("russian")
	if err != nil {
		t.Fatal(err)
	}
	if len(ru.AbbrevTypes) < 1000 {
		t.Fatalf("russian: abbrevs=%d", len(ru.AbbrevTypes))
	}
}
