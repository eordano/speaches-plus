package punkt

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

const ParamsEnv = "NV_PUNKT_DATA"

var curatedAbbrevs = []string{
	"adm", "al", "apr", "approx", "aug", "ave", "b.a", "blvd", "brig", "capt", "cf", "cmdr",
	"co", "col", "corp", "cpl", "dec", "dept", "dr", "e.g", "ed", "est", "et", "etc", "feb",
	"fig", "figs", "fri", "ft", "gen", "gov", "hon", "hr", "i.e", "inc", "jan", "jr", "jul",
	"jun", "lt", "ltd", "m.d", "maj", "mar", "messrs", "mfg", "mgr", "mlle", "mme", "mon",
	"mr", "mrs", "ms", "msgr", "mt", "nov", "oct", "p.m", "ph.d", "pp", "prof", "pvt", "rep",
	"rev", "rd", "sat", "sen", "sep", "sept", "sgt", "sr", "st", "sun", "thu", "thurs", "tue",
	"tues", "u.k", "u.n", "u.s", "u.s.a", "univ", "v", "vol", "vols", "vs", "wed",
}

var (
	englishOnce   sync.Once
	englishParams *Params
)

func curatedParams() *Params {
	p := &Params{
		AbbrevTypes:  make(map[string]bool),
		Collocations: make(map[[2]string]bool),
		SentStarters: make(map[string]bool),
		OrthoContext: make(map[string]uint8),
	}
	for _, a := range curatedAbbrevs {
		p.AbbrevTypes[a] = true
	}
	return p
}

func readLines(dir, name string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return nil, err
	}
	var out []string
	for _, l := range strings.Split(string(raw), "\n") {
		if l != "" {
			out = append(out, l)
		}
	}
	return out, nil
}

func LoadPunktTab(dir string) (*Params, error) {
	p := &Params{
		AbbrevTypes:  make(map[string]bool),
		Collocations: make(map[[2]string]bool),
		SentStarters: make(map[string]bool),
		OrthoContext: make(map[string]uint8),
	}
	lines, err := readLines(dir, "abbrev_types.txt")
	if err != nil {
		return nil, err
	}
	for _, l := range lines {
		p.AbbrevTypes[l] = true
	}
	if lines, err = readLines(dir, "sent_starters.txt"); err != nil {
		return nil, err
	}
	for _, l := range lines {
		p.SentStarters[l] = true
	}
	if lines, err = readLines(dir, "collocations.tab"); err != nil {
		return nil, err
	}
	for _, l := range lines {
		a, b, ok := strings.Cut(l, "\t")
		if !ok {
			return nil, fmt.Errorf("%s/collocations.tab: bad line %q", dir, l)
		}
		p.Collocations[[2]string{a, b}] = true
	}
	if lines, err = readLines(dir, "ortho_context.tab"); err != nil {
		return nil, err
	}
	for _, l := range lines {
		t, f, ok := strings.Cut(l, "\t")
		if !ok {
			return nil, fmt.Errorf("%s/ortho_context.tab: bad line %q", dir, l)
		}
		v, err := strconv.ParseUint(f, 10, 8)
		if err != nil {
			return nil, fmt.Errorf("%s/ortho_context.tab: bad flag %q", dir, l)
		}
		p.OrthoContext[t] = uint8(v)
	}
	return p, nil
}

func Trained(lang string) (*Params, error) {
	root := os.Getenv(ParamsEnv)
	if root == "" {
		return nil, fmt.Errorf("%s unset", ParamsEnv)
	}
	return LoadPunktTab(filepath.Join(root, lang))
}

func EnglishTrained() (*Params, error) {
	p, err := Trained("english")
	if err != nil {
		return nil, err
	}
	for _, a := range curatedAbbrevs {
		p.AbbrevTypes[a] = true
	}
	return p, nil
}

func EnglishParams() *Params {
	englishOnce.Do(func() {
		p, err := EnglishTrained()
		if err != nil {
			fmt.Fprintf(os.Stderr, "punkt: %v; falling back to curated abbreviations only\n", err)
			p = curatedParams()
		}
		englishParams = p
	})
	return englishParams
}

func English() *Segmenter {
	return NewSegmenter(EnglishParams())
}
