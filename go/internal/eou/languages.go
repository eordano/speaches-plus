package eou

import (
	"encoding/json"
	"os"
)

type LanguageConfig struct {
	Threshold float32 `json:"threshold"`
	Name      string  `json:"name,omitempty"`
}

type LanguageTable map[string]LanguageConfig

func (c *LanguageConfig) UnmarshalJSON(data []byte) error {
	type alias LanguageConfig
	var v alias
	if err := json.Unmarshal(data, &v); err == nil {
		*c = LanguageConfig(v)
		return nil
	}
	var f float32
	if err := json.Unmarshal(data, &f); err == nil {
		c.Threshold = f
		return nil
	}
	var s struct {
		Threshold float32 `json:"threshold"`
		Name      string  `json:"name,omitempty"`
	}
	return json.Unmarshal(data, &s)
}

var defaultLanguages = LanguageTable{
	"en": {Threshold: 0.55},
	"es": {Threshold: 0.55},
	"fr": {Threshold: 0.55},
	"de": {Threshold: 0.55},
	"it": {Threshold: 0.55},
	"pt": {Threshold: 0.55},
	"ja": {Threshold: 0.45},
	"zh": {Threshold: 0.45},
	"ko": {Threshold: 0.45},
}

func DefaultLanguages() LanguageTable {
	out := make(LanguageTable, len(defaultLanguages))
	for k, v := range defaultLanguages {
		out[k] = v
	}
	return out
}

func LoadLanguages(path string) (LanguageTable, error) {
	if path == "" {
		return DefaultLanguages(), nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return DefaultLanguages(), err
	}
	tbl := LanguageTable{}
	if err := json.Unmarshal(raw, &tbl); err != nil {
		return DefaultLanguages(), err
	}
	merged := DefaultLanguages()
	for k, v := range tbl {
		merged[k] = v
	}
	return merged, nil
}

func (t LanguageTable) Threshold(lang string) float32 {
	if cfg, ok := t[lang]; ok {
		return cfg.Threshold
	}
	if cfg, ok := t["en"]; ok {
		return cfg.Threshold
	}
	return 0.55
}
