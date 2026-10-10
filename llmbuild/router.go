package llmbuild

import (
	"errors"
	"fmt"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/llmconfig"
)

// Router собирает плечи cfg.Active() и задачи развёрнутого cfg; ошибки всех плеч — вместе, по имени.
func Router(cfg llmconfig.Config, o Options) (*llm.Router, error) {
	r := &llm.Router{
		Providers: map[string]llm.Provider{},
		Speech:    map[string]llm.Transcriber{},
		OCR:       map[string]llm.Recognizer{},
	}

	var errs []error

	for _, name := range cfg.Active() {
		b, err := Arm(name, cfg.Providers[name], o)
		if err != nil {
			errs = append(errs, fmt.Errorf("providers.%s: %w", name, err))

			continue
		}

		switch {
		case b.Provider != nil:
			r.Providers[name] = b.Provider
		case b.Speech != nil:
			r.Speech[name] = b.Speech
		case b.OCR != nil:
			r.OCR[name] = b.OCR
		}
	}

	var err error

	r.Tasks, err = cfg.Routes()
	errs = append(errs, err)

	r.SpeechTasks, err = cfg.SpeechRoutes()
	errs = append(errs, err)

	r.OCRTasks, err = cfg.OCRRoutes()
	errs = append(errs, err)

	if err = errors.Join(errs...); err != nil {
		return nil, err
	}

	return r, nil
}
