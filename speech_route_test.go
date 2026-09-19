package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

// speaker — плечо речи с одним профилем «m» до секунды; отвечает шагами по очереди, потом успехом.
type speaker struct {
	steps []error

	mu    sync.Mutex
	calls int
}

func (*speaker) Name() string { return "fake-asr" }

func (*speaker) SpeechCapabilities(model string) (llm.SpeechCapabilities, bool) {
	return llm.SpeechCapabilities{MaxAudio: time.Second}, model == "m"
}

func (s *speaker) Transcribe(_ context.Context, req llm.SpeechRequest) (llm.Transcript, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.calls++
	if s.calls <= len(s.steps) {
		return llm.Transcript{}, s.steps[s.calls-1]
	}

	return llm.Transcript{
		Text:        "привет " + req.Language,
		RequestID:   "rq",
		AudioMillis: req.Duration().Milliseconds(),
	}, nil
}

func speechRouter(s *speaker, cfg llm.SpeechTaskConfig, obs llm.Observer) *llm.Router {
	return &llm.Router{
		Speech:      map[string]llm.Transcriber{"asr": s},
		SpeechTasks: map[string]llm.SpeechTaskConfig{"dictation": cfg},
		Observe:     obs,
	}
}

// halfSecond — 0,5 с тишины при 16 кГц.
func halfSecond() llm.SpeechInput {
	return llm.SpeechInput{CallID: "c1", SampleRate: 16000, PCM: make([]byte, 16000)}
}

// TestResolveSpeechWithoutSection — речь не объявлена: false без ошибки, чатовый Validate её не требует.
func TestResolveSpeechWithoutSection(t *testing.T) {
	t.Parallel()

	router, _ := textRouter()

	_, ok, err := router.ResolveSpeech("dictation")
	if ok || err != nil {
		t.Fatalf("ResolveSpeech без раздела = %v, %v", ok, err)
	}

	if err = router.Validate(nil); err != nil {
		t.Fatal(err)
	}
}

func TestResolveSpeechRejects(t *testing.T) {
	t.Parallel()

	router := &llm.Router{
		Speech: map[string]llm.Transcriber{"asr": &speaker{}},
		SpeechTasks: map[string]llm.SpeechTaskConfig{
			"a": {Provider: "nope", Model: "m"},
			"b": {Model: "m"},
			"c": {Provider: "asr", Model: "x"},
			"d": {Provider: "asr", Model: "m", Attempts: -1},
			"e": {Provider: "asr", Model: "m", AttemptTimeout: -time.Second},
			"f": {Provider: "asr"},
		},
	}

	wants := map[string]string{
		"a": `speech.a: плечо "nope" не собрано`,
		"b": "speech.b: плечо не задано",
		"c": "speech.c: fake-asr/x: профиль модели не подтверждён",
		"d": "speech.d: отрицательное число попыток",
		"e": "speech.e: отрицательный срок попытки",
		"f": "speech.f: модель не задана",
	}

	err := router.Validate(nil)

	for task, want := range wants {
		if _, ok, resolveErr := router.ResolveSpeech(task); ok || resolveErr == nil ||
			!strings.Contains(resolveErr.Error(), want) {
			t.Errorf("ResolveSpeech(%s) = %v, %v; want %q", task, ok, resolveErr, want)
		}

		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Validate без %q: %v", want, err)
		}
	}
}

// TestSpeechRetriesServerError — 5xx повторяется; отчёты несут задачу речи, запись и известный ноль токенов.
func TestSpeechRetriesServerError(t *testing.T) {
	t.Parallel()

	s := &speaker{steps: []error{&llm.StatusError{Status: 503, RequestID: "r503"}}}
	obs := &recorder{}
	cfg := llm.SpeechTaskConfig{Provider: "asr", Model: "m", Language: "ru-RU", Attempts: 2}

	route, ok, err := speechRouter(s, cfg, obs).ResolveSpeech("dictation")
	if !ok || err != nil {
		t.Fatalf("ResolveSpeech = %v, %v", ok, err)
	}

	res, err := route.Transcribe(context.Background(), halfSecond())
	if err != nil {
		t.Fatal(err)
	}

	if res.Text != "привет ru-RU" || res.Model != "m" || len(res.Attempts) != 2 {
		t.Fatalf("результат %+v", res)
	}

	if len(obs.attempts) != 2 || len(obs.calls) != 1 {
		t.Fatalf("отчётов попыток %d, вызовов %d", len(obs.attempts), len(obs.calls))
	}

	first, call := obs.attempts[0], obs.calls[0]
	if first.Outcome != llm.OutcomeHTTP5xx || first.RequestID != "r503" || first.AudioMillis != 500 {
		t.Fatalf("первая попытка %+v: 5xx мог распознать запись, она в расходе", first)
	}

	if call.Task != "dictation" || call.Provider != "fake-asr" || call.Outcome != llm.OutcomeOK ||
		call.AudioMillis != 1000 || !call.Usage.Known || call.Usage.InputTokens()+call.Usage.OutputTokens() != 0 {
		t.Fatalf("отчёт вызова %+v", call)
	}
}

// TestSpeechDoesNotRetryClientError — 4xx не повторяется и записи в расход не несёт.
func TestSpeechDoesNotRetryClientError(t *testing.T) {
	t.Parallel()

	s := &speaker{steps: []error{&llm.StatusError{Status: 400}}}
	obs := &recorder{}

	route, _, err := speechRouter(s, llm.SpeechTaskConfig{Provider: "asr", Model: "m"}, obs).ResolveSpeech("dictation")
	if err != nil {
		t.Fatal(err)
	}

	_, err = route.Transcribe(context.Background(), halfSecond())

	var call *llm.CallError
	if !errors.As(err, &call) || call.Class != llm.RetryNever || len(call.Attempts) != 1 || s.calls != 1 {
		t.Fatalf("отказ %v, вызовов плеча %d", err, s.calls)
	}

	if obs.attempts[0].AudioMillis != 0 || call.Report.AudioMillis != 0 || call.Report.Outcome != llm.OutcomeError {
		t.Fatalf("отчёт %+v", call.Report)
	}
}

// TestSpeechRejectsLongAudioBeforeProvider — запись сверх профиля до плеча не доходит.
func TestSpeechRejectsLongAudioBeforeProvider(t *testing.T) {
	t.Parallel()

	s := &speaker{}
	obs := &recorder{}

	route, _, err := speechRouter(s, llm.SpeechTaskConfig{Provider: "asr", Model: "m"}, obs).ResolveSpeech("dictation")
	if err != nil {
		t.Fatal(err)
	}

	in := halfSecond()
	in.PCM = make([]byte, 64000)

	_, err = route.Transcribe(context.Background(), in)

	var call *llm.CallError
	if !errors.As(err, &call) || call.Class != llm.RetryNever || call.Report.Outcome != llm.OutcomeBadRequest {
		t.Fatalf("отказ %v", err)
	}

	if s.calls != 0 || len(obs.attempts) != 0 || len(obs.calls) != 1 {
		t.Fatalf("плечо вызвано %d раз, попыток %d", s.calls, len(obs.attempts))
	}
}

func TestSpeechRouteBudget(t *testing.T) {
	t.Parallel()

	cfg := llm.SpeechTaskConfig{Provider: "asr", Model: "m", AttemptTimeout: 20 * time.Second, Attempts: 2}

	route, _, err := speechRouter(&speaker{}, cfg, nil).ResolveSpeech("dictation")
	if err != nil {
		t.Fatal(err)
	}

	want := llm.New(nil, 20*time.Second)
	want.Attempts = 2

	if route.Budget() != want.Budget() {
		t.Fatalf("бюджет %v, ожидался %v", route.Budget(), want.Budget())
	}

	var zero llm.SpeechRoute
	if zero.Budget() != 0 {
		t.Fatal("нулевой маршрут с бюджетом")
	}

	if _, err = zero.Transcribe(context.Background(), halfSecond()); err == nil {
		t.Fatal("нулевой маршрут распознал запись")
	}
}

func TestSpeechDescriptorFingerprint(t *testing.T) {
	t.Parallel()

	a := llm.SpeechDescriptor{Task: "t", Provider: "p1", Kind: "speechkit", Model: "general", Language: "ru-RU"}
	b := a
	b.Provider, b.PricePlan = "p2", "plan"

	if a.Fingerprint() != b.Fingerprint() {
		t.Fatal("имя плеча и тариф поменяли отпечаток")
	}

	b.Language = "en-US"
	if a.Fingerprint() == b.Fingerprint() {
		t.Fatal("язык не вошёл в отпечаток")
	}
}
