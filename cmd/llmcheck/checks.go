//go:build live

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/pricing"
)

const (
	callIDBytes = 8
	imageSide   = 64
	opaque      = 0xff
	// millisSince — граница единиц expires_at, та же, что у адаптера gigachat.
	millisSince = 100_000_000_000
	// minTokenTTL — запас обновления токена у адаптера: токен короче обновлялся бы каждым вызовом.
	minTokenTTL   = 5 * time.Minute
	maxTokenTTL   = 24 * time.Hour
	maxOAuthBody  = 64 << 10
	finishTokens  = 1
	uuidVersion4  = 0x40
	uuidVariant   = 0x80
	uuidLowNibble = 0x0f
	uuidVarMask   = 0x3f
)

// Слова ответов, по которым видно, что модель получила вход, а не выдумала ответ.
const (
	fieldOK      = "ok"
	fieldColor   = "color"
	fieldAnswer  = "answer"
	typeBoolean  = "boolean"
	typeString   = "string"
	typeInteger  = "integer"
	wantTrue     = "true"
	wantProduct  = "391"
	productQuery = "Сколько будет 17 умножить на 23?"
)

type status string

const (
	statusPass status = "pass"
	statusFail status = "fail"
	statusSkip status = "skip"
)

// result — строка протокола.
type result struct {
	check   string
	subject string
	status  status
	notes   []string
	call    *callFacts
}

// callFacts — обезличенное о вызове: без текста запроса, токенов и идентификаторов поставщика.
type callFacts struct {
	model     string
	finish    llm.Finish
	usage     llm.Usage
	attempts  int
	latency   time.Duration
	requestID bool
	cleanup   int
	cost      pricing.Cost
}

// errRequestCap — потолок обращений исчерпан.
var errRequestCap = errors.New("потолок обращений llmcheck исчерпан")

// meter — счётчик обращений на весь прогон.
type meter struct {
	mu    sync.Mutex
	limit int
	used  int
}

func (m *meter) take() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.used >= m.limit {
		return errRequestCap
	}

	m.used++

	return nil
}

// counted — плечо за счётчиком: каждая попытка клиента берёт обращение до сети.
type counted struct {
	llm.Provider

	meter *meter
}

func (c counted) Complete(ctx context.Context, req llm.Request) (llm.Result, error) {
	if err := c.meter.take(); err != nil {
		return llm.Result{}, &llm.RequestError{Message: err.Error(), Err: err}
	}

	return c.Provider.Complete(ctx, req)
}

type runner struct {
	cfg        *llmconfig.Config
	router     *llm.Router
	tasks      []string
	meter      *meter
	maxCost    int64
	spent      map[string]int64
	seen       map[string]string
	oauth      map[string]oauthTarget
	out        io.Writer
	results    []result
	incomplete bool
}

func (r *runner) printf(format string, args ...any) {
	_, _ = fmt.Fprintf(r.out, format, args...)
}

func (r *runner) route(task string) llm.Route {
	route, _ := r.router.Resolve(task)

	return route
}

// runAll — OAuth по плечам GigaChat, затем проверки задач в порядке allChecks.
func (r *runner) runAll(ctx context.Context, checks []string) {
	if slices.Contains(checks, checkOAuth) {
		for _, name := range slices.Sorted(maps.Keys(r.oauth)) {
			r.add(r.guard(ctx, checkOAuth, name, func() result { return r.checkOAuth(ctx, name) }))
		}
	}

	perTask := map[string]func(context.Context, string) result{
		checkModel:     r.checkModel,
		checkVision:    r.checkVision,
		checkReasoning: r.checkReasoning,
		checkFinish:    r.checkFinish,
		checkLimits:    r.checkLimits,
	}

	for _, task := range r.tasks {
		for _, check := range allChecks() {
			fn, ok := perTask[check]
			if !ok || !slices.Contains(checks, check) {
				continue
			}

			r.add(r.guard(ctx, check, task, func() result { return fn(ctx, task) }))
		}
	}
}

// guard — проверка не начинается за потолком расхода или сроком; упёршаяся в потолок обращений — пропуск.
func (r *runner) guard(ctx context.Context, check, subject string, fn func() result) result {
	skip := result{check: check, subject: subject, status: statusSkip}

	switch {
	case ctx.Err() != nil:
		r.incomplete = true
		skip.notes = []string{"срок прогона истёк"}

		return skip
	case r.overBudget():
		r.incomplete = true
		skip.notes = []string{"потолок расхода достигнут"}

		return skip
	}

	res := fn()
	if res.status == statusFail && slices.ContainsFunc(res.notes, isCapNote) {
		r.incomplete = true
		res.status = statusSkip
	}

	return res
}

func isCapNote(note string) bool { return strings.Contains(note, errRequestCap.Error()) }

func (r *runner) add(res result) {
	r.results = append(r.results, res)

	r.printf("- [%s] %s %s", res.status, res.check, res.subject)

	if res.call != nil {
		r.printf(": %s", res.call.line())
	}

	r.printf("\n")

	for _, note := range res.notes {
		r.printf("  - %s\n", note)
	}
}

// overBudget — потолок расхода достигнут: следующая проверка не начнётся.
func (r *runner) overBudget() bool {
	return slices.ContainsFunc(slices.Collect(maps.Values(r.spent)), func(v int64) bool { return v >= r.maxCost })
}

// overspent — потолок перелетел: вызов, начатый под ним, со всеми повторами оплачен целиком.
func (r *runner) overspent() bool {
	return slices.ContainsFunc(slices.Collect(maps.Values(r.spent)), func(v int64) bool { return v > r.maxCost })
}

// dedupe — та же проверка того же маршрута уже прошла у другой задачи: платить второй раз незачем.
func (r *runner) dedupe(check, subject, key string) (result, bool) {
	key = check + "|" + key
	if first, ok := r.seen[key]; ok {
		return result{
			check: check, subject: subject, status: statusSkip,
			notes: []string{"маршрут тот же, что у задачи " + first},
		}, true
	}

	r.seen[key] = subject

	return result{}, false
}

func (r *runner) checkModel(ctx context.Context, task string) result {
	route := r.route(task)
	desc := route.Descriptor()

	if dup, ok := r.dedupe(checkModel, task, desc.Provider+"|"+desc.Fingerprint()); ok {
		return dup
	}

	msgs := []llm.Message{{
		Role: llm.RoleUser,
		Text: prompt(desc.Output.Mode, fieldOK, typeBoolean, "Проверка связи: ответь true."),
	}}
	res, err := route.Chat(ctx, input(desc, msgs, fieldOK, typeBoolean))

	out := r.verdict(checkModel, task, res, err)
	if err == nil {
		isTrue := func(v any) bool { return v == true }
		out.expect(answerMatches(desc.Output.Mode, res.Text, fieldOK, isTrue, wantTrue), "ответ не ok=true")
	}

	return out
}

func (r *runner) checkVision(ctx context.Context, task string) result {
	route := r.route(task)
	desc := route.Descriptor()

	if caps, _ := r.router.Providers[desc.Provider].Capabilities(desc.Model); !caps.Vision {
		return result{
			check:   checkVision,
			subject: task,
			status:  statusSkip,
			notes:   []string{"профиль модели без кадров"},
		}
	}

	if dup, ok := r.dedupe(checkVision, task, desc.Provider+"|"+desc.Fingerprint()); ok {
		return dup
	}

	img, err := redImage()
	if err != nil {
		return result{check: checkVision, subject: task, status: statusFail, notes: []string{err.Error()}}
	}

	msgs := []llm.Message{{
		Role:   llm.RoleUser,
		Text:   prompt(desc.Output.Mode, fieldColor, typeString, "Какого цвета кадр? Назови цвет одним словом."),
		Images: []llm.Image{img},
	}}
	res, err := route.Chat(ctx, input(desc, msgs, fieldColor, typeString))

	out := r.verdict(checkVision, task, res, err)
	if err == nil {
		isRed := func(v any) bool {
			s, ok := v.(string)

			return ok && isWord(s, colorWords()...)
		}
		out.expect(answerMatches(desc.Output.Mode, res.Text, fieldColor, isRed, colorWords()...),
			"цвет кадра не назван: кадр не дошёл до модели")
	}

	return out
}

func (r *runner) checkReasoning(ctx context.Context, task string) result {
	route := r.route(task)
	desc := route.Descriptor()

	if desc.Options.Reasoning == "" {
		return result{
			check:   checkReasoning,
			subject: task,
			status:  statusSkip,
			notes:   []string{"маршрут без reasoning"},
		}
	}

	if dup, ok := r.dedupe(checkReasoning, task, desc.Provider+"|"+desc.Fingerprint()); ok {
		return dup
	}

	msgs := []llm.Message{{Role: llm.RoleUser, Text: prompt(desc.Output.Mode, fieldAnswer, typeInteger, productQuery)}}
	res, err := route.Chat(ctx, input(desc, msgs, fieldAnswer, typeInteger))

	out := r.verdict(checkReasoning, task, res, err)
	if err == nil {
		isProduct := func(v any) bool {
			n, ok := v.(json.Number)

			return ok && n.String() == wantProduct
		}
		out.expect(
			answerMatches(desc.Output.Mode, res.Text, fieldAnswer, isProduct, wantProduct),
			"неверное произведение",
		)

		if res.Usage.Reasoning == 0 {
			out.notes = append(out.notes, "токены рассуждений не сообщены или ноль")
		}
	}

	return out
}

// checkFinish — предел ответа в один токен обязан кончиться исходом truncated с расходом, классом never
// и без повтора: повторы клиенту оставлены, иначе «без повтора» ничего не доказывало бы.
func (r *runner) checkFinish(ctx context.Context, task string) result {
	desc := r.route(task).Descriptor()
	provider := r.router.Providers[desc.Provider]

	if caps, _ := provider.Capabilities(desc.Model); !caps.MaxOutputTokens {
		return result{
			check:   checkFinish,
			subject: task,
			status:  statusSkip,
			notes:   []string{"профиль без предела ответа"},
		}
	}

	if dup, ok := r.dedupe(checkFinish, task, desc.Provider+"|"+desc.Model); ok {
		return dup
	}

	res, err := llm.New(provider, 0).Chat(ctx, llm.Request{
		CallID:   callID(),
		Task:     task,
		Model:    desc.Model,
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "Перечисли по порядку все дни недели."}},
		Options:  llm.Options{MaxOutputTokens: finishTokens},
	})

	out := r.facts(task, res, err)
	out.check, out.subject = checkFinish, task

	var call *llm.CallError

	switch {
	case errors.Is(err, llm.ErrTruncated) && errors.As(err, &call):
		out.status = statusPass
		out.expect(call.Class == llm.RetryNever, fmt.Sprintf("класс обрезанного %s, ждали never", call.Class))
		out.expect(out.call.usage.Known && out.call.usage.OutputTokens() > 0, "у обрезанного ответа расход неизвестен")
	case err != nil:
		out.status = statusFail
		out.notes = append(out.notes, "ждали truncated: "+failNote(err))
	default:
		out.status = statusFail
		out.notes = append(out.notes, fmt.Sprintf("ответ в %d токен не обрезан", finishTokens))
	}

	if out.call.attempts > 1 {
		out.fail("обрезанный ответ повторён")
	}

	return out
}

// checkLimits — предельное число кадров профиля одним вызовом: загрузка принята, уборка прошла.
func (r *runner) checkLimits(ctx context.Context, task string) result {
	desc := r.route(task).Descriptor()
	provider := r.router.Providers[desc.Provider]

	caps, _ := provider.Capabilities(desc.Model)
	if !caps.Vision || caps.MaxImagesPerRequest == 0 {
		return result{
			check:   checkLimits,
			subject: task,
			status:  statusSkip,
			notes:   []string{"у профиля нет предела кадров"},
		}
	}

	if dup, ok := r.dedupe(checkLimits, task, desc.Provider+"|"+desc.Model); ok {
		return dup
	}

	img, err := redImage()
	if err != nil {
		return result{check: checkLimits, subject: task, status: statusFail, notes: []string{err.Error()}}
	}

	perMessage := caps.MaxImagesPerMessage
	if perMessage == 0 {
		perMessage = caps.MaxImagesPerRequest
	}

	var msgs []llm.Message

	for left := caps.MaxImagesPerRequest; left > 0; left -= perMessage {
		msgs = append(
			msgs,
			llm.Message{Role: llm.RoleUser, Images: slices.Repeat([]llm.Image{img}, min(left, perMessage))},
		)
	}

	msgs[len(msgs)-1].Text = "Сколько кадров выше? Ответь числом."

	client := llm.New(provider, 0)
	client.Attempts = 1

	res, err := client.Chat(ctx, llm.Request{CallID: callID(), Task: task, Model: desc.Model, Messages: msgs})

	out := r.verdict(checkLimits, task, res, err)
	out.notes = append(out.notes, fmt.Sprintf("кадров %d, на сообщение до %d", caps.MaxImagesPerRequest, perMessage))

	return out
}

// checkOAuth — срок токена GigaChat в тех единицах, что ждёт адаптер, и не короче его запаса обновления.
func (r *runner) checkOAuth(ctx context.Context, name string) result {
	out := result{check: checkOAuth, subject: name, status: statusPass}

	if err := r.meter.take(); err != nil {
		out.fail(err.Error())

		return out
	}

	expiresAt, err := r.fetchExpiry(ctx, r.oauth[name])
	if err != nil {
		out.fail(err.Error())

		return out
	}

	unit, expires := "секунды", time.Unix(expiresAt, 0)
	if expiresAt > millisSince {
		unit, expires = "миллисекунды", time.UnixMilli(expiresAt)
	}

	ttl := time.Until(expires)
	out.notes = append(out.notes, fmt.Sprintf("expires_at в единицах «%s», срок %s", unit, ttl.Round(time.Minute)))

	switch {
	case ttl <= minTokenTTL:
		out.fail("срок токена не длиннее запаса обновления адаптера " + minTokenTTL.String())
	case ttl > maxTokenTTL:
		out.fail("срок токена больше суток: единицы expires_at разобраны неверно")
	}

	return out
}

func (r *runner) fetchExpiry(ctx context.Context, t oauthTarget) (int64, error) {
	body := strings.NewReader(url.Values{"scope": {t.scope}}.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, body)
	if err != nil {
		return 0, fmt.Errorf("запрос OAuth: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Rquid", uuid4())
	req.Header.Set("Authorization", "Basic "+t.key)

	resp, err := t.http.Do(req)
	if err != nil {
		return 0, fmt.Errorf("запрос OAuth: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("OAuth: HTTP %d", resp.StatusCode)
	}

	var env struct {
		AccessToken string `json:"access_token"`
		ExpiresAt   int64  `json:"expires_at"`
	}

	if err = json.NewDecoder(io.LimitReader(resp.Body, maxOAuthBody)).Decode(&env); err != nil {
		return 0, fmt.Errorf("ответ OAuth: %w", err)
	}

	if env.AccessToken == "" || env.ExpiresAt <= 0 {
		return 0, errors.New("ответ OAuth без токена или срока")
	}

	return env.ExpiresAt, nil
}

// verdict — вызов удался, расход полон и ненулев, генерация кончилась stop, уборка прошла.
func (r *runner) verdict(check, task string, res llm.Result, err error) result {
	out := r.facts(task, res, err)
	out.check, out.subject, out.status = check, task, statusPass

	if err != nil {
		out.fail(failNote(err))

		return out
	}

	u := out.call.usage
	out.expect(u.Known, "расход сообщён не полностью")
	out.expect(u.InputTokens() > 0 && u.OutputTokens() > 0, "нулевой расход у удавшейся генерации")
	out.expect(res.Finish.Kind == llm.FinishStop, fmt.Sprintf("finish_reason %q, ждали stop", res.Finish.Raw))
	out.expect(out.call.cleanup == 0, "файлы попытки не убраны")

	if res.Model == "" {
		out.notes = append(out.notes, "модель в ответе не названа")
	}

	return out
}

// facts — сведения о вызове из результата или отказа, с оценкой стоимости по тарифу задачи.
func (r *runner) facts(task string, res llm.Result, err error) result {
	f := &callFacts{
		model:     res.Model,
		finish:    res.Finish,
		usage:     res.Report.Usage,
		latency:   res.Latency,
		requestID: res.RequestID != "",
	}
	attempts := res.Attempts

	var call *llm.CallError
	if errors.As(err, &call) {
		f.model, f.finish, f.usage, f.latency = call.Report.Model, call.Finish, call.Report.Usage, call.Latency
		attempts = call.Attempts
	}

	f.attempts = len(attempts)

	for _, a := range attempts {
		f.requestID = f.requestID || a.RequestID != ""

		if a.Cleanup != nil {
			f.cleanup++
		}
	}

	f.cost = pricing.Cost{Status: pricing.StatusUnknown}

	if plan := r.cfg.Tasks[task].PricePlan; plan != "" {
		plans, _ := r.cfg.PricePlans(plan)
		f.cost = pricing.EstimateCall(attempts, plans...)
	}

	if f.cost.Status == pricing.StatusEstimated || f.cost.Status == pricing.StatusPartial {
		r.spent[f.cost.Currency] += f.cost.AmountMicro
	}

	return result{call: f}
}

func (res *result) fail(note string) {
	res.status = statusFail
	res.notes = append(res.notes, note)
}

func (res *result) expect(ok bool, note string) {
	if !ok {
		res.fail(note)
	}
}

func (f *callFacts) line() string {
	u := f.usage
	known := "полный"

	if !u.Known {
		known = "неполный"
	}

	parts := []string{
		"модель " + orDash(f.model),
		fmt.Sprintf("finish %s (%s)", orDash(string(f.finish.Kind)), orDash(f.finish.Raw)),
		fmt.Sprintf("расход %s: вход %d, кеш %d, выход %d, рассуждения %d",
			known, u.BillableInput, u.CachedInput, u.Output, u.Reasoning),
	}

	if len(u.Raw) > 0 {
		raw := make([]string, 0, len(u.Raw))
		for _, key := range slices.Sorted(maps.Keys(u.Raw)) {
			raw = append(raw, fmt.Sprintf("%s=%d", key, u.Raw[key]))
		}

		parts = append(parts, "сырые "+strings.Join(raw, " "))
	}

	parts = append(parts,
		fmt.Sprintf("попыток %d", f.attempts),
		f.latency.Round(time.Millisecond).String(),
		fmt.Sprintf("request id %t", f.requestID),
		fmt.Sprintf("стоимость %s", costLine(f.cost)),
	)

	return strings.Join(parts, ", ")
}

func costLine(c pricing.Cost) string {
	if c.Status == pricing.StatusUnknown || c.Status == pricing.StatusFree {
		return string(c.Status)
	}

	return fmt.Sprintf("%d мк. %s по %s (%s)", c.AmountMicro, c.Currency, c.Revision, c.Status)
}

// input — вход маршрута: форма его же, схема из одного поля — только у маршрута со схемой.
func input(desc llm.Descriptor, msgs []llm.Message, field, typ string) llm.Input {
	in := llm.Input{CallID: callID(), Messages: msgs}

	if desc.Output.Mode == llm.ModeSchema {
		in.Schema = json.RawMessage(fmt.Sprintf(
			`{"type":"object","properties":{%q:{"type":%q}},"required":[%q],"additionalProperties":false}`,
			field, typ, field,
		))
	}

	return in
}

func prompt(mode llm.Mode, field, typ, question string) string {
	switch mode {
	case llm.ModeJSON:
		return question + " Ответь JSON-объектом с единственным полем " + field + " типа " + typ + "."
	case llm.ModeSchema:
		return question + " Ответ — в поле " + field + "."
	case llm.ModeText:
		return question + " Ответь коротко."
	default:
		return question
	}
}

// answerMatches — у JSON-форм поле ответа проходит want, у текста текст целиком — одно из слов.
func answerMatches(mode llm.Mode, text, field string, want func(any) bool, words ...string) bool {
	if mode != llm.ModeJSON && mode != llm.ModeSchema {
		return isWord(text, words...)
	}

	v, ok := answerField(text, field, mode == llm.ModeSchema)

	return ok && want(v)
}

// answerField — поле единственного JSON-объекта ответа, числа — json.Number; exact требует объект ровно
// из этого поля, как велит схема проверки с additionalProperties=false.
func answerField(text, field string, exact bool) (any, bool) {
	dec := json.NewDecoder(strings.NewReader(text))
	dec.UseNumber()

	var obj map[string]any
	if dec.Decode(&obj) != nil {
		return nil, false
	}

	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, false
	}

	v, ok := obj[field]
	if !ok || (exact && len(obj) != 1) {
		return nil, false
	}

	return v, true
}

// isWord — ответ без регистра и обрамляющей пунктуации совпадает со словом целиком: подстрока
// приняла бы «not red» и «некрасный».
func isWord(s string, words ...string) bool {
	s = strings.ToLower(strings.TrimFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }))

	return slices.Contains(words, s)
}

// colorWords — названия цвета кадра проверки vision.
func colorWords() []string {
	return []string{"красный", "красная", "красное", "red"}
}

// failNote — отказ словами протокола: класс, исход, фаза и код. Текст ошибки поставщика не печатается —
// в нём бывает текст ответа модели.
func failNote(err error) string {
	var call *llm.CallError

	switch {
	case errors.Is(err, errRequestCap):
		return errRequestCap.Error()
	case !errors.As(err, &call):
		return "отказ вне вызова плеча"
	}

	parts := []string{
		"класс " + string(call.Class),
		"исход " + orDash(call.Report.Outcome),
		"фаза " + orDash(string(call.Phase)),
	}

	if call.Status != 0 {
		parts = append(parts, fmt.Sprintf("HTTP %d", call.Status))
	}

	return "отказ: " + strings.Join(parts, ", ")
}

// redImage — сплошной красный PNG: цвет однозначен, а назвать его без кадра нельзя.
func redImage() (llm.Image, error) {
	img := image.NewRGBA(image.Rect(0, 0, imageSide, imageSide))
	for y := range imageSide {
		for x := range imageSide {
			img.Set(x, y, color.RGBA{R: opaque, A: opaque})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return llm.Image{}, fmt.Errorf("кадр проверки: %w", err)
	}

	return llm.Image{MIME: "image/png", Data: buf.Bytes()}, nil
}

func callID() string {
	b := make([]byte, callIDBytes)
	_, _ = rand.Read(b)

	return "llmcheck-" + hex.EncodeToString(b)
}

// uuid4 — RqUID запроса OAuth.
func uuid4() string {
	b := make([]byte, 16) //nolint:mnd // размер UUID
	_, _ = rand.Read(b)
	b[6] = b[6]&uuidLowNibble | uuidVersion4
	b[8] = b[8]&uuidVarMask | uuidVariant

	h := hex.EncodeToString(b)

	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}
