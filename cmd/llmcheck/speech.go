//go:build live

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/pricing"
	"github.com/lllypuk/llm/salutespeech"
	"github.com/lllypuk/llm/yandex"
)

const (
	cmdSpeech       = "speech"
	defaultMaxFiles = 10
	wavExt          = ".wav"
	textExt         = ".txt"
	textPerm        = 0o600
	microPerUnit    = 1_000_000
)

// Заголовок WAV: RIFF/WAVE, чанки fmt и data; принимается только LPCM 16 бит моно.
const (
	riffHeader    = 12
	chunkHeader   = 8
	fmtMinSize    = 16
	formatPCM     = 1
	monoChannels  = 1
	bitsPerSample = 16
	// chunkAlign — тело чанка нечётной длины добивается байтом до чётной.
	chunkAlign = 2
)

type speechOptions struct {
	config      string
	dir         string
	task        string
	maxFiles    int
	maxRequests int
	maxCost     int64
	timeout     time.Duration
}

// clip — запись из каталога без контейнера WAV.
type clip struct {
	name string
	path string
	rate int
	pcm  []byte
}

func speechMain(ctx context.Context, args []string, stdout, stderr io.Writer, lookup llmconfig.Lookup) int {
	o, err := parseSpeechFlags(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return exitOK
	}

	if err != nil {
		return speechUsage(stderr, err)
	}

	return runSpeech(ctx, o, stdout, stderr, lookup)
}

func parseSpeechFlags(args []string, stderr io.Writer) (speechOptions, error) {
	fs := flag.NewFlagSet("llmcheck speech", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var o speechOptions

	fs.StringVar(&o.config, "config", "", "файл маршрутов llmconfig; секреты — из окружения")
	fs.StringVar(&o.dir, "dir", "", "каталог с записями *.wav: LPCM 16 бит моно; текст ляжет рядом в *.txt")
	fs.StringVar(&o.task, "task", "", "задача раздела speech; пусто — единственная объявленная")
	fs.IntVar(&o.maxFiles, "max-files", defaultMaxFiles, "потолок числа записей: больше — отказ до первого вызова")
	fs.IntVar(&o.maxRequests, "max-requests", defaultMaxRequests, "потолок обращений к плечу, включая повторы")
	fs.Int64Var(&o.maxCost, "max-cost", defaultMaxCost,
		"порог оценённого расхода в микроединицах валюты: следующая запись за ним не отправляется")
	fs.DurationVar(&o.timeout, "timeout", defaultTimeout, "срок прогона целиком")

	if err := fs.Parse(args); err != nil {
		return speechOptions{}, err
	}

	var errs []error

	if fs.NArg() > 0 {
		errs = append(errs, fmt.Errorf("лишние аргументы: %s", strings.Join(fs.Args(), " ")))
	}

	if o.config == "" {
		errs = append(errs, errors.New("-config: файл маршрутов не задан"))
	}

	if o.dir == "" {
		errs = append(errs, errors.New("-dir: каталог записей не задан"))
	}

	if o.maxFiles <= 0 {
		errs = append(errs, errors.New("-max-files: потолок не положителен"))
	}

	if o.maxRequests <= 0 {
		errs = append(errs, errors.New("-max-requests: потолок не положителен"))
	}

	if o.maxCost <= 0 {
		errs = append(errs, errors.New("-max-cost: потолок не положителен"))
	}

	if o.timeout <= 0 {
		errs = append(errs, errors.New("-timeout: срок не положителен"))
	}

	return o, errors.Join(errs...)
}

func runSpeech(ctx context.Context, o speechOptions, stdout, stderr io.Writer, lookup llmconfig.Lookup) int {
	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()

	data, err := os.ReadFile(o.config)
	if err != nil {
		return speechUsage(stderr, err)
	}

	clips, err := loadClips(o.dir, o.maxFiles)
	if err != nil {
		return speechUsage(stderr, err)
	}

	r, route, err := prepareSpeech(data, o, lookup)
	if err != nil {
		return speechUsage(stderr, err)
	}

	r.out = stdout
	r.speechHeader(data, o, route, len(clips))
	r.transcribeAll(ctx, route, clips)

	return r.footer()
}

func speechUsage(stderr io.Writer, err error) int {
	_, _ = fmt.Fprintln(stderr, "llmcheck speech:", err)

	return exitUsage
}

// loadClips читает все записи до первого вызова: негодная или лишняя отбивает прогон, не потратив денег.
func loadClips(dir string, maxFiles int) ([]clip, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var names []string

	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), wavExt) {
			names = append(names, e.Name())
		}
	}

	switch {
	case len(names) == 0:
		return nil, fmt.Errorf("-dir: в %s нет записей *%s", dir, wavExt)
	case len(names) > maxFiles:
		return nil, fmt.Errorf("-max-files: записей %d, потолок %d", len(names), maxFiles)
	}

	clips := make([]clip, 0, len(names))

	var errs []error

	for _, name := range names {
		path := filepath.Join(dir, name)

		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			errs = append(errs, readErr)

			continue
		}

		c, wavErr := parseWAV(raw)
		if wavErr != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, wavErr))

			continue
		}

		c.name, c.path = name, path
		clips = append(clips, c)
	}

	return clips, errors.Join(errs...)
}

// parseWAV снимает контейнер: частоту берёт из fmt, сэмплы — из data; посторонние чанки пропускает.
func parseWAV(b []byte) (clip, error) {
	if len(b) < riffHeader || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return clip{}, errors.New("не RIFF/WAVE")
	}

	rate := 0

	for off := riffHeader; off+chunkHeader <= len(b); {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+chunkHeader]))
		body := off + chunkHeader

		if size > len(b)-body {
			return clip{}, fmt.Errorf("чанк %q усечён", id)
		}

		chunk := b[body : body+size]

		switch id {
		case "fmt ":
			var err error
			if rate, err = parseFmt(chunk); err != nil {
				return clip{}, err
			}
		case "data":
			if rate == 0 {
				return clip{}, errors.New("чанк data раньше fmt")
			}

			return clip{rate: rate, pcm: chunk}, nil
		}

		off = body + size + size%chunkAlign
	}

	return clip{}, errors.New("нет чанка data")
}

func parseFmt(chunk []byte) (int, error) {
	if len(chunk) < fmtMinSize {
		return 0, errors.New("чанк fmt короче 16 байт")
	}

	format := binary.LittleEndian.Uint16(chunk[0:2])
	channels := binary.LittleEndian.Uint16(chunk[2:4])
	rate := int(binary.LittleEndian.Uint32(chunk[4:8]))
	bits := binary.LittleEndian.Uint16(chunk[14:16])

	switch {
	case format != formatPCM:
		return 0, fmt.Errorf("формат %d, нужен PCM", format)
	case channels != monoChannels:
		return 0, fmt.Errorf("каналов %d, нужен один", channels)
	case bits != bitsPerSample:
		return 0, fmt.Errorf("%d бит на сэмпл, нужно 16", bits)
	case rate <= 0:
		return 0, errors.New("частота не задана")
	}

	return rate, nil
}

// prepareSpeech оставляет в конфиге одну задачу речи и собирает её плечо за общим счётчиком.
func prepareSpeech(data []byte, o speechOptions, lookup llmconfig.Lookup) (*runner, llm.SpeechRoute, error) {
	cfg, err := llmconfig.Parse(data)
	if err != nil {
		return nil, llm.SpeechRoute{}, err
	}

	task := o.task
	if task == "" {
		declared := slices.Sorted(maps.Keys(cfg.Speech))
		if len(declared) != 1 {
			return nil, llm.SpeechRoute{}, fmt.Errorf("-task: в разделе speech задач %d, нужна одна", len(declared))
		}

		task = declared[0]
	}

	if _, ok := cfg.Speech[task]; !ok {
		return nil, llm.SpeechRoute{}, fmt.Errorf("-task: задача речи %q не объявлена", task)
	}

	cfg.Tasks = nil
	maps.DeleteFunc(cfg.Speech, func(name string, _ llmconfig.SpeechTask) bool { return name != task })

	if err = cfg.Expand(lookup); err != nil {
		return nil, llm.SpeechRoute{}, err
	}

	routes, err := cfg.SpeechRoutes()
	if err != nil {
		return nil, llm.SpeechRoute{}, err
	}

	r := &runner{
		cfg:     cfg,
		tasks:   []string{task},
		meter:   &meter{limit: o.maxRequests},
		maxCost: o.maxCost,
		spent:   map[string]int64{},
		router:  &llm.Router{Speech: map[string]llm.Transcriber{}, SpeechTasks: routes},
	}

	for _, name := range cfg.Active() {
		t, buildErr := buildSpeech(cfg.Providers[name])
		if buildErr != nil {
			return nil, llm.SpeechRoute{}, fmt.Errorf("providers.%s: %w", name, buildErr)
		}

		r.router.Speech[name] = countedSpeech{Transcriber: t, meter: r.meter}
	}

	route, _, err := r.router.ResolveSpeech(task)
	if err != nil {
		return nil, llm.SpeechRoute{}, err
	}

	return r, route, nil
}

func buildSpeech(p llmconfig.Provider) (llm.Transcriber, error) {
	pool, err := loadCA(p.CAFile)
	if err != nil {
		return nil, err
	}

	client := trusting(pool)

	switch p.Kind {
	case llmconfig.KindSpeechKit:
		s, buildErr := yandex.NewSpeech(yandex.SpeechConfig{
			Endpoint:    p.Endpoint,
			Folder:      p.Folder,
			Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
			HTTP:        client,
		})
		if buildErr != nil {
			return nil, buildErr
		}

		return s, nil
	case llmconfig.KindSaluteSpeech:
		s, buildErr := salutespeech.New(salutespeech.Config{
			OAuthEndpoint:    p.OAuthEndpoint,
			APIEndpoint:      p.Endpoint,
			AuthorizationKey: p.Auth.AuthorizationKey.Value(),
			Scope:            p.Scope,
			CA:               pool,
			HTTP:             client,
		})
		if buildErr != nil {
			return nil, buildErr
		}

		return s, nil
	default:
		return nil, fmt.Errorf("вид плеча %q не распознаёт речь", p.Kind)
	}
}

// countedSpeech — плечо речи за счётчиком: каждая попытка берёт обращение до сети.
type countedSpeech struct {
	llm.Transcriber

	meter *meter
}

func (c countedSpeech) Transcribe(ctx context.Context, req llm.SpeechRequest) (llm.Transcript, error) {
	if err := c.meter.take(); err != nil {
		return llm.Transcript{}, &llm.RequestError{Message: err.Error(), Err: err}
	}

	return c.Transcriber.Transcribe(ctx, req)
}

// speechHeader — шапка протокола без секретов, путей и текста записей.
func (r *runner) speechHeader(data []byte, o speechOptions, route llm.SpeechRoute, files int) {
	sum := sha256.Sum256(data)
	desc := route.Descriptor()

	r.printf("# llmcheck speech %s\n\n", time.Now().Format(time.DateOnly))
	r.printf("- сборка: %s\n", buildLine())
	r.printf("- конфиг: sha256 %s\n", hex.EncodeToString(sum[:configHashBytes]))
	r.printf("- потолки: записей %d, обращений %d, расхода %d мк. на валюту\n", o.maxFiles, o.maxRequests, o.maxCost)
	r.printf("- плечо: %s\n", providerLine(desc.Provider, r.cfg.Providers[desc.Provider]))
	r.printf("- задача: %s: %s/%s, язык %s, бюджет %s, тариф %s\n",
		desc.Task, desc.Provider, desc.Model, orDash(desc.Language), route.Budget(), orDash(desc.PricePlan))
	r.printf("- записей: %d\n\n## Записи\n\n", files)
}

func (r *runner) transcribeAll(ctx context.Context, route llm.SpeechRoute, clips []clip) {
	var plans []pricing.PricePlan
	if name := route.Descriptor().PricePlan; name != "" {
		plans, _ = r.cfg.PricePlans(name)
	}

	for _, c := range clips {
		r.add(r.guard(ctx, cmdSpeech, c.name, func() result { return r.transcribe(ctx, route, plans, c) }))
	}
}

// transcribe — одна запись; текст ложится файлом рядом с ней, в протокол не попадает.
func (r *runner) transcribe(ctx context.Context, route llm.SpeechRoute, plans []pricing.PricePlan, c clip) result {
	res := result{check: cmdSpeech, subject: c.name, status: statusPass}

	out, err := route.Transcribe(ctx, llm.SpeechInput{CallID: callID(), SampleRate: c.rate, PCM: c.pcm})
	attempts, latency, audio := out.Attempts, out.Latency, out.Report.AudioMillis

	var call *llm.CallError
	if errors.As(err, &call) {
		attempts, latency, audio = call.Attempts, call.Latency, call.Report.AudioMillis
	}

	cost := pricing.Cost{Status: pricing.StatusUnknown}
	if len(plans) > 0 {
		cost = pricing.EstimateCall(attempts, plans...)
	}

	if cost.Status == pricing.StatusEstimated || cost.Status == pricing.StatusPartial {
		r.spent[cost.Currency] += cost.AmountMicro
	}

	res.line = fmt.Sprintf("задержка %s, секунд %.2f, попыток %d, стоимость %s",
		latency.Round(time.Millisecond), time.Duration(audio*int64(time.Millisecond)).Seconds(), len(attempts),
		moneyLine(cost))

	if err != nil {
		res.fail(failNote(err))

		return res
	}

	if out.Text == "" {
		res.notes = append(res.notes, "пустой текст")
	}

	if writeErr := os.WriteFile(strings.TrimSuffix(c.path, filepath.Ext(c.path))+textExt,
		[]byte(out.Text), textPerm); writeErr != nil {
		res.fail("текст не записан: " + writeErr.Error())
	}

	return res
}

// moneyLine — оценка в единицах валюты, а не в микроединицах: протокол сверяют с тарифом глазами.
func moneyLine(c pricing.Cost) string {
	if c.Status == pricing.StatusUnknown || c.Status == pricing.StatusFree {
		return string(c.Status)
	}

	return fmt.Sprintf("%d.%06d %s по %s (%s)",
		c.AmountMicro/microPerUnit, c.AmountMicro%microPerUnit, c.Currency, c.Revision, c.Status)
}
