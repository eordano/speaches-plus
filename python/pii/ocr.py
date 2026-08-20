from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class OcrToken:
    text: str
    start: int
    end_exclusive: int
    left: int
    top: int
    right: int
    bottom: int

@dataclass
class OcrResult:
    text: str
    tokens: list[OcrToken]

def _try_pytesseract(image: np.ndarray) -> OcrResult:
    import pytesseract
    from PIL import Image

    pil_image = Image.fromarray(image)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    tokens: list[OcrToken] = []
    offset = 0
    text_parts: list[str] = []

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        word = data["text"][i].strip()
        if not word:
            continue

        start = offset
        end_exclusive = offset + len(word)

        tokens.append(OcrToken(
            text=word,
            start=start,
            end_exclusive=end_exclusive,
            left=data["left"][i],
            top=data["top"][i],
            right=data["left"][i] + data["width"][i],
            bottom=data["top"][i] + data["height"][i],
        ))

        if text_parts:
            offset += 1
        text_parts.append(word)
        offset = end_exclusive

    full_text = " ".join(text_parts)
    return OcrResult(text=full_text, tokens=tokens)

def _try_paddleocr(image: np.ndarray) -> OcrResult:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(image, cls=True)

    tokens: list[OcrToken] = []
    offset = 0
    text_parts: list[str] = []

    if result and result[0]:
        for line in result[0]:
            box_points = line[0]
            word = line[1][0].strip()
            if not word:
                continue

            xs = [p[0] for p in box_points]
            ys = [p[1] for p in box_points]
            left = int(min(xs))
            top = int(min(ys))
            right = int(max(xs))
            bottom = int(max(ys))

            start = offset
            end_exclusive = offset + len(word)

            tokens.append(OcrToken(
                text=word,
                start=start,
                end_exclusive=end_exclusive,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            ))

            if text_parts:
                offset += 1
            text_parts.append(word)
            offset = end_exclusive

    full_text = " ".join(text_parts)
    return OcrResult(text=full_text, tokens=tokens)

def _try_easyocr(image: np.ndarray) -> OcrResult:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=True)
    results = reader.readtext(image)

    tokens: list[OcrToken] = []
    offset = 0
    text_parts: list[str] = []

    for (box_points, word, _conf) in results:
        word = word.strip()
        if not word:
            continue

        xs = [p[0] for p in box_points]
        ys = [p[1] for p in box_points]
        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs))
        bottom = int(max(ys))

        start = offset
        end_exclusive = offset + len(word)

        tokens.append(OcrToken(
            text=word,
            start=start,
            end_exclusive=end_exclusive,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        ))

        if text_parts:
            offset += 1
        text_parts.append(word)
        offset = end_exclusive

    full_text = " ".join(text_parts)
    return OcrResult(text=full_text, tokens=tokens)

def run_ocr(image: np.ndarray) -> OcrResult:
    try:
        return _try_pytesseract(image)
    except ImportError:
        pass

    try:
        return _try_paddleocr(image)
    except ImportError:
        pass

    try:
        return _try_easyocr(image)
    except ImportError:
        pass

    raise ImportError(
        "No OCR backend available. Install one of: pytesseract, paddleocr, easyocr"
    )
