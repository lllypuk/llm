package llm

import (
	"context"
	"time"
)

// bytesPerSample — 16-битный LPCM, единственный формат контракта речи.
const bytesPerSample = 2

// Transcriber — плечо распознавания речи. Чатового [Provider] не реализует: probe
// и Complete у него бессмысленны.
type Transcriber interface {
	Name() string
	SpeechCapabilities(model string) (SpeechCapabilities, bool)
	Transcribe(ctx context.Context, req SpeechRequest) (Transcript, error)
}

// SpeechCapabilities — что подтверждено у модели распознавания; MaxAudio ноль — предела нет.
type SpeechCapabilities struct {
	MaxAudio time.Duration
}

// SpeechRequest — один вызов распознавания: сырые сэмплы LPCM, 16 бит, моно, little-endian.
type SpeechRequest struct {
	Model      string
	Language   string
	SampleRate int
	PCM        []byte
}

// Duration — длительность записи по числу сэмплов; при негодной частоте ноль.
func (r SpeechRequest) Duration() time.Duration {
	if r.SampleRate <= 0 {
		return 0
	}

	return time.Duration(len(r.PCM)/bytesPerSample) * time.Second / time.Duration(r.SampleRate)
}

// Validate отбивает запрос, который плечо не примет, до сети. Ошибка — [*RequestError].
func (r SpeechRequest) Validate(maxAudio time.Duration) error {
	var msg string

	switch {
	case r.Model == "":
		msg = "модель не задана"
	case r.SampleRate <= 0:
		msg = "частота дискретизации не задана"
	case len(r.PCM) == 0:
		msg = "пустая запись"
	case len(r.PCM)%bytesPerSample != 0:
		msg = "нечётная длина PCM при 16 битах на сэмпл"
	case maxAudio > 0 && r.Duration() > maxAudio:
		msg = "запись длиннее " + maxAudio.String()
	default:
		return nil
	}

	return &RequestError{Message: msg}
}

// Transcript — распознанный текст; пустой Text — успех, в записи не расслышали речи.
type Transcript struct {
	Text        string
	Model       string
	RequestID   string
	AudioMillis int64
}
