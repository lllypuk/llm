//go:build live

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/pricing"
	"github.com/lllypuk/llm/yandex"
)

const cmdOCR = "ocr"

// frame — кадр из каталога; MIME — по содержимому, а не по расширению.
type frame struct {
	name  string
	path  string
	mime  string
	image []byte
}

func ocrMain(ctx context.Context, args []string, stdout, stderr io.Writer, lookup llmconfig.Lookup) int {
	o, err := parseDirFlags(cmdOCR, "каталог с кадрами *.jpg и *.png; текст ляжет рядом в *.txt", args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return exitOK
	}

	if err != nil {
		return ocrUsage(stderr, err)
	}

	return runOCR(ctx, o, stdout, stderr, lookup)
}

func runOCR(ctx context.Context, o dirOptions, stdout, stderr io.Writer, lookup llmconfig.Lookup) int {
	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()

	data, err := os.ReadFile(o.config)
	if err != nil {
		return ocrUsage(stderr, err)
	}

	frames, err := loadFrames(o.dir, o.maxFiles)
	if err != nil {
		return ocrUsage(stderr, err)
	}

	r, route, err := prepareOCR(data, o, lookup)
	if err != nil {
		return ocrUsage(stderr, err)
	}

	r.out = stdout
	r.ocrHeader(data, o, route, len(frames))
	r.recognizeAll(ctx, route, frames)

	return r.footer()
}

func ocrUsage(stderr io.Writer, err error) int {
	_, _ = fmt.Fprintln(stderr, "llmcheck ocr:", err)

	return exitUsage
}

// loadFrames читает все кадры до первого вызова: негодный или лишний отбивает прогон, не потратив денег.
func loadFrames(dir string, maxFiles int) ([]frame, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var names []string

	for _, e := range entries {
		switch strings.ToLower(filepath.Ext(e.Name())) {
		case ".jpg", ".jpeg", ".png":
			if !e.IsDir() {
				names = append(names, e.Name())
			}
		}
	}

	switch {
	case len(names) == 0:
		return nil, fmt.Errorf("-dir: в %s нет кадров *.jpg и *.png", dir)
	case len(names) > maxFiles:
		return nil, fmt.Errorf("-max-files: кадров %d, потолок %d", len(names), maxFiles)
	}

	if clashErr := textClash(names); clashErr != nil {
		return nil, clashErr
	}

	frames := make([]frame, 0, len(names))

	var errs []error

	for _, name := range names {
		path := filepath.Join(dir, name)

		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			errs = append(errs, readErr)

			continue
		}

		mime := http.DetectContentType(raw)
		if mime != llm.MIMEJPEG && mime != llm.MIMEPNG {
			errs = append(errs, fmt.Errorf("%s: содержимое %s, нужен JPEG или PNG", name, mime))

			continue
		}

		frames = append(frames, frame{name: name, path: path, mime: mime, image: raw})
	}

	return frames, errors.Join(errs...)
}

// prepareOCR собирает плечо выбранной задачи OCR за общим счётчиком.
//
//nolint:dupl // близнец prepareSpeech: типы плеча и маршрута разные
func prepareOCR(data []byte, o dirOptions, lookup llmconfig.Lookup) (*runner, llm.OCRRoute, error) {
	cfg, task, err := narrowConfig(data, lookup, cmdOCR, "OCR", o.task,
		func(c *llmconfig.Config) *map[string]llmconfig.OCRTask { return &c.OCR })
	if err != nil {
		return nil, llm.OCRRoute{}, err
	}

	routes, err := cfg.OCRRoutes()
	if err != nil {
		return nil, llm.OCRRoute{}, err
	}

	r := dirRunner(cfg, task, o, &llm.Router{OCR: map[string]llm.Recognizer{}, OCRTasks: routes})

	err = buildActive(cfg, buildOCR, func(name string, rec llm.Recognizer) {
		r.router.OCR[name] = countedOCR{Recognizer: rec, meter: r.meter}
	})
	if err != nil {
		return nil, llm.OCRRoute{}, err
	}

	route, _, err := r.router.ResolveOCR(task)
	if err != nil {
		return nil, llm.OCRRoute{}, err
	}

	return r, route, nil
}

func buildOCR(p llmconfig.Provider) (llm.Recognizer, error) {
	if p.Kind != llmconfig.KindVisionOCR {
		return nil, fmt.Errorf("вид плеча %q не распознаёт текст", p.Kind)
	}

	pool, err := loadCA(p.CAFile)
	if err != nil {
		return nil, err
	}

	rec, err := yandex.NewOCR(yandex.OCRConfig{
		Endpoint:    p.Endpoint,
		Folder:      p.Folder,
		Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
		HTTP:        trusting(pool),
	})
	if err != nil {
		return nil, err
	}

	return rec, nil
}

// countedOCR — плечо OCR за счётчиком: каждая попытка берёт обращение до сети.
type countedOCR struct {
	llm.Recognizer

	meter *meter
}

func (c countedOCR) Recognize(ctx context.Context, req llm.OCRRequest) (llm.OCRText, error) {
	if err := c.meter.take(); err != nil {
		return llm.OCRText{}, &llm.RequestError{Message: err.Error(), Err: err}
	}

	return c.Recognizer.Recognize(ctx, req)
}

// ocrHeader — шапка протокола без секретов, путей и текста кадров.
func (r *runner) ocrHeader(data []byte, o dirOptions, route llm.OCRRoute, files int) {
	sum := sha256.Sum256(data)
	desc := route.Descriptor()

	r.printf("# llmcheck ocr %s\n\n", time.Now().Format(time.DateOnly))
	r.printf("- сборка: %s\n", buildLine())
	r.printf("- конфиг: sha256 %s\n", hex.EncodeToString(sum[:configHashBytes]))
	r.printf("- потолки: кадров %d, обращений %d, расхода %d мк. на валюту\n", o.maxFiles, o.maxRequests, o.maxCost)
	r.printf("- плечо: %s\n", providerLine(desc.Provider, r.cfg.Providers[desc.Provider]))
	r.printf("- задача: %s: %s/%s, языки %s, бюджет %s, тариф %s\n",
		desc.Task, desc.Provider, desc.Model, strings.Join(desc.Languages, ","), route.Budget(), orDash(desc.PricePlan))
	r.printf("- кадров: %d\n\n## Кадры\n\n", files)
}

func (r *runner) recognizeAll(ctx context.Context, route llm.OCRRoute, frames []frame) {
	var plans []pricing.PricePlan
	if name := route.Descriptor().PricePlan; name != "" {
		plans, _ = r.cfg.PricePlans(name)
	}

	for _, f := range frames {
		r.add(r.guard(ctx, cmdOCR, f.name, func() result { return r.recognize(ctx, route, plans, f) }))
	}
}

// recognize — один кадр; текст ложится файлом рядом с ним, в протокол не попадает.
func (r *runner) recognize(ctx context.Context, route llm.OCRRoute, plans []pricing.PricePlan, f frame) result {
	res := result{check: cmdOCR, subject: f.name, status: statusPass}

	out, err := route.Recognize(ctx, llm.OCRInput{CallID: callID(), Image: f.image, MIME: f.mime})
	attempts, latency, pages := out.Attempts, out.Latency, out.Report.Pages

	var call *llm.CallError
	if errors.As(err, &call) {
		attempts, latency, pages = call.Attempts, call.Latency, call.Report.Pages
	}

	cost := pricing.Cost{Status: pricing.StatusUnknown}
	if len(plans) > 0 {
		cost = pricing.EstimateCall(attempts, plans...)
	}

	if cost.Status == pricing.StatusEstimated || cost.Status == pricing.StatusPartial {
		r.spent[cost.Currency] += cost.AmountMicro
	}

	res.line = fmt.Sprintf("задержка %s, страниц %d, попыток %d, стоимость %s",
		latency.Round(time.Millisecond), pages, len(attempts), moneyLine(cost))

	if err != nil {
		res.failOn(err, failNote(err))

		return res
	}

	if out.Text == "" {
		res.notes = append(res.notes, "пустой текст")
	}

	if writeErr := os.WriteFile(strings.TrimSuffix(f.path, filepath.Ext(f.path))+textExt,
		[]byte(out.Text), textPerm); writeErr != nil {
		res.fail("текст не записан: " + writeErr.Error())
	}

	return res
}
