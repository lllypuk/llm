package llm

import (
	"context"
	"strconv"
)

// MIME-типы кадра, которые принимает контракт OCR; PDF растеризует потребитель.
const (
	MIMEJPEG = "image/jpeg"
	MIMEPNG  = "image/png"
)

// Recognizer — плечо распознавания текста на изображении. Чатового [Provider] не реализует.
type Recognizer interface {
	Name() string
	OCRCapabilities(model string) (OCRCapabilities, bool)
	Recognize(ctx context.Context, req OCRRequest) (OCRText, error)
}

// OCRCapabilities — что подтверждено у модели OCR; MaxBytes ноль — предела нет.
type OCRCapabilities struct {
	MaxBytes int
}

// OCRRequest — один кадр на распознавание: JPEG или PNG целиком.
type OCRRequest struct {
	Model     string
	Image     []byte
	MIME      string
	Languages []string
}

// Validate отбивает запрос, который плечо не примет, до сети. Ошибка — [*RequestError].
func (r OCRRequest) Validate(maxBytes int) error {
	var msg string

	switch {
	case r.Model == "":
		msg = msgNoModel
	case len(r.Image) == 0:
		msg = "пустой кадр"
	case r.MIME != MIMEJPEG && r.MIME != MIMEPNG:
		msg = "формат кадра " + strconv.Quote(r.MIME) + " не поддержан, нужен " + MIMEJPEG + " или " + MIMEPNG
	case maxBytes > 0 && len(r.Image) > maxBytes:
		msg = "кадр больше " + strconv.Itoa(maxBytes) + " байт"
	default:
		return nil
	}

	return &RequestError{Message: msg}
}

// OCRText — распознанный текст кадра; пустой Text — успех, текста на кадре нет.
type OCRText struct {
	Text      string
	Model     string
	RequestID string
}
