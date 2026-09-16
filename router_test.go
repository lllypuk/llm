package llm_test

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

// routed — плечо с профилем по модели, запоминающее запросы; ответ повторяет CallID.
type routed struct {
	name     string
	profiles map[string]llm.Capabilities
	err      error

	mu   sync.Mutex
	reqs []llm.Request
}

func (p *routed) Name() string { return p.name }

func (p *routed) Capabilities(model string) (llm.Capabilities, bool) {
	caps, ok := p.profiles[model]

	return caps, ok
}

func (p *routed) Complete(_ context.Context, req llm.Request) (llm.Result, error) {
	p.mu.Lock()
	p.reqs = append(p.reqs, req)
	p.mu.Unlock()

	if p.err != nil {
		return llm.Result{}, p.err
	}

	return llm.Result{
		Text:      req.CallID,
		RequestID: req.CallID,
		Usage:     llm.Usage{BillableInput: 1, Output: 1, Known: true},
	}, nil
}

func (p *routed) requests() []llm.Request {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]llm.Request(nil), p.reqs...)
}

// cleaning — плечо с уборкой после срока попытки.
type cleaning struct{ routed }

func (*cleaning) AttemptOverhead() time.Duration { return 7 * time.Second }

func allCaps() llm.Capabilities {
	return llm.Capabilities{
		Vision: true, JSON: true, Schema: true, Strict: true, Temperature: true, Reasoning: true, MaxOutputTokens: true,
	}
}

func textRouter() (*llm.Router, *routed) {
	p := &routed{name: "fake", profiles: map[string]llm.Capabilities{"m": allCaps(), "m2": allCaps()}}

	return &llm.Router{
		Providers: map[string]llm.Provider{"local": p},
		Tasks: map[string]llm.TaskConfig{
			"ask": {
				Provider:       "local",
				Model:          "m",
				Output:         llm.Output{Mode: llm.ModeSchema, Name: "answer", Strict: true},
				Options:        llm.Options{Temperature: llm.Ptr(0.2), MaxOutputTokens: 100},
				AttemptTimeout: time.Second,
				Attempts:       1,
				PricePlan:      "free",
			},
		},
	}, p
}

func text(s string) []llm.Message { return []llm.Message{{Role: llm.RoleUser, Text: s}} }

// TestRouteIsSnapshot — маршрут разрешён один раз: правка Router, его плеч и копии дескриптора
// после Resolve не меняет ни дескриптор, ни запрос к плечу.
func TestRouteIsSnapshot(t *testing.T) {
	t.Parallel()

	router, p := textRouter()

	route, err := router.Resolve("ask")
	if err != nil {
		t.Fatal(err)
	}

	cfg := router.Tasks["ask"]
	cfg.Model = "m2"
	*cfg.Options.Temperature = 1
	router.Tasks["ask"] = cfg
	router.Providers["local"] = &routed{name: "other"}

	d := route.Descriptor()
	*d.Options.Temperature = 5

	_, err = route.Chat(context.Background(), llm.Input{CallID: "c", Messages: text("x"), Schema: []byte(`{}`)})
	if err != nil {
		t.Fatal(err)
	}

	got := route.Descriptor()
	if got.Model != "m" || *got.Options.Temperature != 0.2 || got.Kind != "fake" || got.Provider != "local" ||
		got.Task != "ask" || got.PricePlan != "free" {
		t.Fatalf("дескриптор изменился: %+v", got)
	}

	reqs := p.requests()
	if len(reqs) != 1 {
		t.Fatalf("запросов к плечу %d", len(reqs))
	}

	r := reqs[0]
	if r.Model != "m" || r.Task != "ask" || r.CallID != "c" || *r.Options.Temperature != 0.2 ||
		r.Options.MaxOutputTokens != 100 || r.Output.Mode != llm.ModeSchema || r.Output.Name != "answer" ||
		!r.Output.Strict || string(r.Output.Schema) != `{}` {
		t.Fatalf("запрос не по маршруту: %+v", r)
	}
}

// TestRouterValidateCollects — неизвестные задача, плечо и профиль, лишняя задача и несовместимые
// возможности отбиваются Validate до вызова, все сразу и с путём.
func TestRouterValidateCollects(t *testing.T) {
	t.Parallel()

	text := llm.Capabilities{JSON: true, Schema: true}
	named := llm.Capabilities{JSON: true, Schema: true, SchemaName: true}
	vision := llm.Capabilities{Vision: true, MaxImagesPerMessage: 1, MaxImagesPerRequest: 10}
	p := &routed{name: "fake", profiles: map[string]llm.Capabilities{"text": text, "named": named, "vision": vision}}

	router := &llm.Router{
		Providers: map[string]llm.Provider{"p": p},
		Tasks: map[string]llm.TaskConfig{
			"a_ghost":   {Provider: "missing", Model: "text"},
			"b_profile": {Provider: "p", Model: "unknown"},
			"c_vision":  {Provider: "p", Model: "text"},
			"d_strict":  {Provider: "p", Model: "text", Output: llm.Output{Mode: llm.ModeSchema, Strict: true}},
			"e_name":    {Provider: "p", Model: "named", Output: llm.Output{Mode: llm.ModeSchema}},
			"f_schema": {
				Provider: "p",
				Model:    "text",
				Output:   llm.Output{Mode: llm.ModeSchema, Schema: []byte(`{}`)},
			},
			"g_images":   {Provider: "p", Model: "vision"},
			"h_attempts": {Provider: "p", Model: "text", Attempts: -1},
			"i_extra":    {Provider: "p", Model: "text"},
			"j_ok":       {Provider: "p", Model: "vision"},
		},
	}

	err := router.Validate(map[string]llm.Needs{
		"a_ghost":    {},
		"b_profile":  {},
		"c_vision":   {ImagesPerMessage: 1, ImagesPerRequest: 1},
		"d_strict":   {},
		"e_name":     {},
		"f_schema":   {},
		"g_images":   {ImagesPerMessage: 2, ImagesPerRequest: 2},
		"h_attempts": {},
		"j_ok":       {ImagesPerMessage: 1, ImagesPerRequest: 10},
		"z_unknown":  {},
	})
	if err == nil {
		t.Fatal("ожидался отказ")
	}

	for _, want := range []string{
		`tasks.a_ghost: плечо "missing" не собрано`,
		"tasks.b_profile: fake/unknown: профиль модели не подтверждён",
		"tasks.c_vision: fake/text: кадры не поддержаны",
		"tasks.d_strict: fake/text: strict не поддержан",
		"tasks.e_name: fake/named: схема без имени",
		"tasks.f_schema: схема в конфиге",
		"tasks.g_images: fake/vision: кадров в сообщении больше 1",
		"tasks.h_attempts: отрицательное число попыток",
		"tasks.i_extra: задача не нужна потребителю",
		"tasks.z_unknown: задача не объявлена",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("нет %q в\n%v", want, err)
		}
	}

	if strings.Contains(err.Error(), "j_ok") {
		t.Errorf("годная задача отбита: %v", err)
	}

	if len(p.requests()) != 0 {
		t.Fatal("Validate ходил к плечу")
	}

	if _, resolveErr := router.Resolve("z_unknown"); resolveErr == nil {
		t.Fatal("неизвестная задача разрешилась")
	}
}

// TestRouteChatRejectsInputConflict — форма и схема входа, расходящиеся с маршрутом, — отказ never
// до плеча с отчётом наблюдателю, а не молча принятое значение маршрута.
func TestRouteChatRejectsInputConflict(t *testing.T) {
	t.Parallel()

	router, p := textRouter()
	router.Tasks["plain"] = llm.TaskConfig{Provider: "local", Model: "m"}
	obs := &recorder{}
	router.Observe = obs

	ask, err := router.Resolve("ask")
	if err != nil {
		t.Fatal(err)
	}

	plain, err := router.Resolve("plain")
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string]struct {
		route llm.Route
		in    llm.Input
	}{
		"форма": {
			ask,
			llm.Input{CallID: "c1", Messages: text("x"), Mode: llm.ModeJSON, Schema: []byte(`{}`)},
		},
		"схема без schema":   {plain, llm.Input{CallID: "c2", Messages: text("x"), Schema: []byte(`{}`)}},
		"json у текстовой":   {plain, llm.Input{CallID: "c3", Messages: text("x"), Mode: llm.ModeJSON}},
		"schema без схемы":   {ask, llm.Input{CallID: "c4", Messages: text("x")}},
		"нулевой маршрут":    {llm.Route{}, llm.Input{CallID: "c5", Messages: text("x")}},
		"текстовая по форме": {plain, llm.Input{CallID: "ok", Messages: text("x"), Mode: llm.ModeText}},
	}

	for name, tc := range cases {
		_, chatErr := tc.route.Chat(context.Background(), tc.in)
		if tc.in.CallID == "ok" {
			if chatErr != nil {
				t.Errorf("%s: %v", name, chatErr)
			}

			continue
		}

		if call := callError(t, chatErr); call.Class != llm.RetryNever {
			t.Errorf("%s: класс %s", name, call.Class)
		}
	}

	if reqs := p.requests(); len(reqs) != 1 || reqs[0].CallID != "ok" {
		t.Fatalf("к плечу дошло лишнее: %+v", reqs)
	}

	obs.mu.Lock()
	defer obs.mu.Unlock()

	outcomes := map[string]string{}
	for _, c := range obs.calls {
		outcomes[c.CallID] = c.Outcome + "/" + c.Task
	}

	if outcomes["c1"] != llm.OutcomeBadRequest+"/ask" || outcomes["c2"] != llm.OutcomeBadRequest+"/plain" ||
		outcomes["c4"] != llm.OutcomeBadRequest+"/ask" {
		t.Fatalf("отчёты наблюдателю: %v", outcomes)
	}
}

// TestRouteHasNoFallback — плечо задачи отказало ключом: отказ уходит вызывающему, второе плечо с той же
// моделью не зовётся.
func TestRouteHasNoFallback(t *testing.T) {
	t.Parallel()

	broken := &routed{
		name:     "broken",
		profiles: map[string]llm.Capabilities{"m": allCaps()},
		err:      &llm.StatusError{Status: 401},
	}
	spare := &routed{name: "spare", profiles: map[string]llm.Capabilities{"m": allCaps()}}

	router := &llm.Router{
		Providers: map[string]llm.Provider{"broken": broken, "spare": spare},
		Tasks: map[string]llm.TaskConfig{
			"ask":   {Provider: "broken", Model: "m", Attempts: 3},
			"other": {Provider: "spare", Model: "m"},
		},
	}

	route, err := router.Resolve("ask")
	if err != nil {
		t.Fatal(err)
	}

	_, err = route.Chat(context.Background(), llm.Input{Messages: text("x")})
	if call := callError(t, err); call.Class != llm.RetryNeedsConfiguration || call.Provider != "broken" {
		t.Fatalf("отказ %+v", call)
	}

	if len(broken.requests()) != 1 || len(spare.requests()) != 0 {
		t.Fatalf("вызовы: плечо задачи %d, запасное %d", len(broken.requests()), len(spare.requests()))
	}
}

// TestRouteBudgetsAreSeparate — у каждой задачи свой бюджет: срок и попытки её конфига плюс уборка
// плеча на каждую попытку.
func TestRouteBudgetsAreSeparate(t *testing.T) {
	t.Parallel()

	local := &routed{name: "local", profiles: map[string]llm.Capabilities{"m": allCaps()}}
	cloud := &cleaning{routed{name: "cloud", profiles: map[string]llm.Capabilities{"m": allCaps()}}}

	router := &llm.Router{
		Providers: map[string]llm.Provider{"local": local, "cloud": cloud},
		Tasks: map[string]llm.TaskConfig{
			"fast": {Provider: "local", Model: "m", AttemptTimeout: time.Second, Attempts: 1},
			"slow": {Provider: "cloud", Model: "m", AttemptTimeout: 20 * time.Second, Attempts: 2},
		},
	}

	fast, err := router.Resolve("fast")
	if err != nil {
		t.Fatal(err)
	}

	slow, err := router.Resolve("slow")
	if err != nil {
		t.Fatal(err)
	}

	if fast.Budget() != time.Second {
		t.Fatalf("бюджет fast %v", fast.Budget())
	}

	want := llm.New(cloud, 20*time.Second)
	want.Attempts = 2

	if got := slow.Budget(); got != want.Budget()+2*7*time.Second {
		t.Fatalf("бюджет slow %v, ожидался %v", got, want.Budget()+14*time.Second)
	}

	if (llm.Route{}).Budget() != 0 {
		t.Fatal("нулевой маршрут с бюджетом")
	}
}

// TestRouteConcurrentReportsDoNotMix — конкурентные вызовы одного маршрута несут свои отчёты.
func TestRouteConcurrentReportsDoNotMix(t *testing.T) {
	t.Parallel()

	router, _ := textRouter()
	router.Tasks["plain"] = llm.TaskConfig{Provider: "local", Model: "m", Attempts: 1}

	route, err := router.Resolve("plain")
	if err != nil {
		t.Fatal(err)
	}

	const n = 64

	errs := make(chan error, n)

	var wg sync.WaitGroup

	for i := range n {
		wg.Go(func() {
			id := strconv.Itoa(i)

			res, chatErr := route.Chat(context.Background(), llm.Input{CallID: id, Messages: text(id)})

			switch {
			case chatErr != nil:
				errs <- chatErr
			case res.Text != id || res.Report.CallID != id || len(res.Attempts) != 1 ||
				res.Attempts[0].CallID != id || res.Attempts[0].RequestID != id:
				errs <- errors.New("отчёт чужого вызова у " + id)
			}
		})
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

// TestDescriptorFingerprint — отпечаток меняют модель, форма, опции и части потребителя; имя плеча
// в конфиге и тариф — нет.
func TestDescriptorFingerprint(t *testing.T) {
	t.Parallel()

	router, _ := textRouter()

	route, err := router.Resolve("ask")
	if err != nil {
		t.Fatal(err)
	}

	base := route.Descriptor()
	same := base
	same.Provider = "renamed"
	same.PricePlan = "paid"

	if base.Fingerprint("prompt") != same.Fingerprint("prompt") {
		t.Fatal("имя плеча или тариф изменили отпечаток")
	}

	changed := map[string]llm.Descriptor{}

	for name, edit := range map[string]func(*llm.Descriptor){
		"модель":      func(d *llm.Descriptor) { d.Model = "m2" },
		"адаптер":     func(d *llm.Descriptor) { d.Kind = "other" },
		"ревизия":     func(d *llm.Descriptor) { d.Revision = "r2" },
		"strict":      func(d *llm.Descriptor) { d.Output.Strict = false },
		"температура": func(d *llm.Descriptor) { d.Options.Temperature = llm.Ptr(0.3) },
		"без темп.":   func(d *llm.Descriptor) { d.Options.Temperature = nil },
	} {
		d := route.Descriptor()
		edit(&d)
		changed[name] = d
	}

	for name, d := range changed {
		if d.Fingerprint("prompt") == base.Fingerprint("prompt") {
			t.Errorf("%s не изменил отпечаток", name)
		}
	}

	if base.Fingerprint("prompt") == base.Fingerprint("prompt2") {
		t.Fatal("части потребителя не входят в отпечаток")
	}
}
