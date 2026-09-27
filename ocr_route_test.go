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

// reader — плечо OCR с одним профилем «page» до 16 байт; отвечает шагами по очереди, потом успехом.
type reader struct {
	steps []error

	mu    sync.Mutex
	calls int
}

func (*reader) Name() string { return "fake-ocr" }

func (*reader) OCRCapabilities(model string) (llm.OCRCapabilities, bool) {
	return llm.OCRCapabilities{MaxBytes: 16}, model == "page"
}

func (r *reader) Recognize(_ context.Context, req llm.OCRRequest) (llm.OCRText, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.calls++
	if r.calls <= len(r.steps) {
		return llm.OCRText{}, r.steps[r.calls-1]
	}

	return llm.OCRText{Text: "чек " + strings.Join(req.Languages, ","), RequestID: "rq"}, nil
}

func ocrRouter(r *reader, cfg llm.OCRTaskConfig, obs llm.Observer) *llm.Router {
	return &llm.Router{
		OCR:      map[string]llm.Recognizer{"vision": r},
		OCRTasks: map[string]llm.OCRTaskConfig{"frame": cfg},
		Observe:  obs,
	}
}

func frame() llm.OCRInput {
	return llm.OCRInput{CallID: "c1", Image: []byte("jpeg"), MIME: llm.MIMEJPEG}
}

// TestResolveOCRWithoutSection — OCR не объявлен: false без ошибки, чатовый Validate его не требует.
func TestResolveOCRWithoutSection(t *testing.T) {
	t.Parallel()

	router, _ := textRouter()

	_, ok, err := router.ResolveOCR("frame")
	if ok || err != nil {
		t.Fatalf("ResolveOCR без раздела = %v, %v", ok, err)
	}

	if err = router.Validate(nil); err != nil {
		t.Fatal(err)
	}
}

func TestResolveOCRRejects(t *testing.T) {
	t.Parallel()

	ru := []string{"ru"}
	router := &llm.Router{
		OCR: map[string]llm.Recognizer{"vision": &reader{}},
		OCRTasks: map[string]llm.OCRTaskConfig{
			"a": {Provider: "nope", Model: "page", Languages: ru},
			"b": {Model: "page", Languages: ru},
			"c": {Provider: "vision", Model: "x", Languages: ru},
			"d": {Provider: "vision", Model: "page", Languages: ru, Attempts: -1},
			"e": {Provider: "vision", Model: "page", Languages: ru, AttemptTimeout: -time.Second},
			"f": {Provider: "vision", Languages: ru},
			"g": {Provider: "vision", Model: "page"},
			"h": {Provider: "vision", Model: "page", Languages: []string{"ru", ""}},
		},
	}

	wants := map[string]string{
		"a": `ocr.a: плечо "nope" не собрано`,
		"b": "ocr.b: плечо не задано",
		"c": "ocr.c: fake-ocr/x: профиль модели не подтверждён",
		"d": "ocr.d: отрицательное число попыток",
		"e": "ocr.e: отрицательный срок попытки",
		"f": "ocr.f: модель не задана",
		"g": "ocr.g: языки не заданы",
		"h": "ocr.h: пустой язык",
	}

	err := router.Validate(nil)

	for task, want := range wants {
		if _, ok, resolveErr := router.ResolveOCR(task); ok || resolveErr == nil ||
			!strings.Contains(resolveErr.Error(), want) {
			t.Errorf("ResolveOCR(%s) = %v, %v; want %q", task, ok, resolveErr, want)
		}

		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Validate без %q: %v", want, err)
		}
	}
}

// TestOCRRetries — 5xx и 429 повторяются; страница в расходе у 5xx и успеха, у 429 — нет.
func TestOCRRetries(t *testing.T) {
	t.Parallel()

	r := &reader{steps: []error{
		&llm.StatusError{Status: 503, RequestID: "r503"},
		&llm.StatusError{Status: 429, RequestID: "r429"},
	}}
	obs := &recorder{}
	cfg := llm.OCRTaskConfig{Provider: "vision", Model: "page", Languages: []string{"ru", "en"}, Attempts: 3}

	route, ok, err := ocrRouter(r, cfg, obs).ResolveOCR("frame")
	if !ok || err != nil {
		t.Fatalf("ResolveOCR = %v, %v", ok, err)
	}

	res, err := route.Recognize(context.Background(), frame())
	if err != nil {
		t.Fatal(err)
	}

	if res.Text != "чек ru,en" || res.Model != "page" || res.RequestID != "rq" || len(res.Attempts) != 3 {
		t.Fatalf("результат %+v", res)
	}

	if len(obs.attempts) != 3 || len(obs.calls) != 1 {
		t.Fatalf("отчётов попыток %d, вызовов %d", len(obs.attempts), len(obs.calls))
	}

	first, second, call := obs.attempts[0], obs.attempts[1], obs.calls[0]
	if first.Outcome != llm.OutcomeHTTP5xx || first.RequestID != "r503" || first.Pages != 1 {
		t.Fatalf("первая попытка %+v: 5xx мог распознать кадр, он в расходе", first)
	}

	if second.RequestID != "r429" || second.Pages != 0 {
		t.Fatalf("вторая попытка %+v: 429 отклонён до работы плеча", second)
	}

	if call.Task != "frame" || call.Provider != "fake-ocr" || call.Outcome != llm.OutcomeOK || call.Pages != 2 ||
		!call.Usage.Known || call.Usage.InputTokens()+call.Usage.OutputTokens() != 0 || call.AudioMillis != 0 {
		t.Fatalf("отчёт вызова %+v", call)
	}
}

// TestOCRDoesNotRetryClientError — 4xx не повторяется и страницы в расход не несёт.
func TestOCRDoesNotRetryClientError(t *testing.T) {
	t.Parallel()

	r := &reader{steps: []error{&llm.StatusError{Status: 400}}}
	obs := &recorder{}
	cfg := llm.OCRTaskConfig{Provider: "vision", Model: "page", Languages: []string{"ru"}}

	route, _, err := ocrRouter(r, cfg, obs).ResolveOCR("frame")
	if err != nil {
		t.Fatal(err)
	}

	_, err = route.Recognize(context.Background(), frame())

	var call *llm.CallError
	if !errors.As(err, &call) || call.Class != llm.RetryNever || len(call.Attempts) != 1 || r.calls != 1 {
		t.Fatalf("отказ %v, вызовов плеча %d", err, r.calls)
	}

	if obs.attempts[0].Pages != 0 || call.Report.Pages != 0 || call.Report.Outcome != llm.OutcomeError {
		t.Fatalf("отчёт %+v", call.Report)
	}
}

// TestOCRPagesByFailure — отказ плеча до сети и 4xx страницы не несут, 5xx и истёкший срок — несут.
func TestOCRPagesByFailure(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		err       error
		pages     int
		requestID string
	}{
		"запрос не собран": {&llm.RequestError{Message: "адрес"}, 0, ""},
		"4xx":              {&llm.StatusError{Status: 400}, 0, ""},
		"5xx":              {&llm.StatusError{Status: 503}, 1, ""},
		"срок":             {context.DeadlineExceeded, 1, ""},
		"сеть":             {errors.New("connection refused"), 1, ""},
		"негодный ответ":   {&llm.ResponseError{RequestID: "rid"}, 1, "rid"},
	} {
		r := &reader{steps: []error{tc.err}}
		obs := &recorder{}
		cfg := llm.OCRTaskConfig{Provider: "vision", Model: "page", Languages: []string{"ru"}, Attempts: 1}

		route, _, err := ocrRouter(r, cfg, obs).ResolveOCR("frame")
		if err != nil {
			t.Fatal(err)
		}

		_, err = route.Recognize(context.Background(), frame())

		var call *llm.CallError
		if !errors.As(err, &call) || r.calls != 1 {
			t.Fatalf("%s: отказ %v, вызовов плеча %d", name, err, r.calls)
		}

		if obs.attempts[0].RequestID != tc.requestID {
			t.Errorf("%s: RequestID попытки %q, ждали %q", name, obs.attempts[0].RequestID, tc.requestID)
		}

		if obs.attempts[0].Pages != tc.pages || call.Report.Pages != tc.pages {
			t.Errorf("%s: страниц в попытке %d, в вызове %d, ждали %d",
				name, obs.attempts[0].Pages, call.Report.Pages, tc.pages)
		}
	}
}

// TestOCRRejectsBeforeProvider — кадр сверх профиля и чужой MIME до плеча не доходят.
func TestOCRRejectsBeforeProvider(t *testing.T) {
	t.Parallel()

	for name, in := range map[string]llm.OCRInput{
		"большой": {Image: make([]byte, 17), MIME: llm.MIMEJPEG},
		"pdf":     {Image: []byte("pdf"), MIME: "application/pdf"},
	} {
		r := &reader{}
		obs := &recorder{}
		cfg := llm.OCRTaskConfig{Provider: "vision", Model: "page", Languages: []string{"ru"}}

		route, _, err := ocrRouter(r, cfg, obs).ResolveOCR("frame")
		if err != nil {
			t.Fatal(err)
		}

		_, err = route.Recognize(context.Background(), in)

		var call *llm.CallError
		if !errors.As(err, &call) || call.Class != llm.RetryNever || call.Report.Outcome != llm.OutcomeBadRequest {
			t.Fatalf("%s: отказ %v", name, err)
		}

		if r.calls != 0 || len(obs.attempts) != 0 || len(obs.calls) != 1 {
			t.Fatalf("%s: плечо вызвано %d раз, попыток %d", name, r.calls, len(obs.attempts))
		}
	}
}

func TestOCRRouteBudget(t *testing.T) {
	t.Parallel()

	cfg := llm.OCRTaskConfig{
		Provider: "vision", Model: "page", Languages: []string{"ru"}, AttemptTimeout: 20 * time.Second, Attempts: 2,
	}

	route, _, err := ocrRouter(&reader{}, cfg, nil).ResolveOCR("frame")
	if err != nil {
		t.Fatal(err)
	}

	want := llm.New(nil, 20*time.Second)
	want.Attempts = 2

	if route.Budget() != want.Budget() {
		t.Fatalf("бюджет %v, ожидался %v", route.Budget(), want.Budget())
	}

	var zero llm.OCRRoute
	if zero.Budget() != 0 {
		t.Fatal("нулевой маршрут с бюджетом")
	}

	if _, err = zero.Recognize(context.Background(), frame()); err == nil {
		t.Fatal("нулевой маршрут распознал кадр")
	}
}

// TestOCRRouteIsImmutable — правка языков в конфиге после Resolve маршрут не меняет.
func TestOCRRouteIsImmutable(t *testing.T) {
	t.Parallel()

	langs := []string{"ru"}
	router := ocrRouter(&reader{}, llm.OCRTaskConfig{Provider: "vision", Model: "page", Languages: langs}, nil)

	route, _, err := router.ResolveOCR("frame")
	if err != nil {
		t.Fatal(err)
	}

	langs[0] = "en"
	route.Descriptor().Languages[0] = "de"

	if got := route.Descriptor().Languages[0]; got != "ru" {
		t.Fatalf("язык маршрута %q", got)
	}
}

func TestOCRDescriptorFingerprint(t *testing.T) {
	t.Parallel()

	a := llm.OCRDescriptor{Task: "t", Provider: "p1", Kind: "visionocr", Model: "page", Languages: []string{"ru", "en"}}
	b := a
	b.Provider, b.PricePlan = "p2", "plan"

	if a.Fingerprint() != b.Fingerprint() {
		t.Fatal("имя плеча и тариф поменяли отпечаток")
	}

	b.Languages = []string{"en", "ru"}
	if a.Fingerprint() == b.Fingerprint() {
		t.Fatal("порядок языков не вошёл в отпечаток")
	}

	b.Languages = []string{"ru"}
	if a.Fingerprint("x") == b.Fingerprint("en", "x") {
		t.Fatal("граница языков и частей потребителя неоднозначна")
	}
}
