package pii

type Rect struct {
	Left   int `json:"left"`
	Top    int `json:"top"`
	Right  int `json:"right"`
	Bottom int `json:"bottom"`
}

type OCRToken struct {
	Start        int  `json:"start"`
	EndExclusive int  `json:"endExclusive"`
	Rect         Rect `json:"rect"`
}

type OCRResult struct {
	Text   string
	Tokens []OCRToken
}
