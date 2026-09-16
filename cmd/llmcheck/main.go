//go:build live

// Команда llmcheck — живые проверки плеч по файлу маршрутов llmconfig. Ходит в сеть и тратит деньги
// аккаунта: потолки обращений и расхода стоят всегда, протокол без секретов и идентификаторов — в stdout.
package main

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/gigachat"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/ollama"
	"github.com/lllypuk/llm/yandex"
)

// Коды выхода: нарушение контракта отличается от прогона, остановленного потолком.
const (
	exitOK         = 0
	exitViolation  = 1
	exitUsage      = 2
	exitIncomplete = 3
)

const (
	defaultMaxRequests = 20
	defaultMaxCost     = 1_000_000
	defaultTimeout     = 10 * time.Minute
	configHashBytes    = 6
)

// Проверки.
const (
	checkOAuth     = "oauth"
	checkModel     = "model"
	checkVision    = "vision"
	checkReasoning = "reasoning"
	checkFinish    = "finish"
	checkLimits    = "limits"
)

// allChecks — порядок прогона.
func allChecks() []string {
	return []string{checkOAuth, checkModel, checkVision, checkReasoning, checkFinish, checkLimits}
}

// defaultChecks — без limits: она шлёт предельное число кадров одним вызовом.
func defaultChecks() []string {
	return []string{checkOAuth, checkModel, checkVision, checkReasoning, checkFinish}
}

type options struct {
	config      string
	tasks       []string
	checks      []string
	maxRequests int
	maxCost     int64
	timeout     time.Duration
}

func main() {
	opts, err := parseFlags(os.Args[1:], os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		os.Exit(exitOK)
	}

	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "llmcheck:", err)
		os.Exit(exitUsage)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	code := run(ctx, opts, os.Stdout, os.Stderr, os.LookupEnv)

	stop()
	os.Exit(code)
}

func parseFlags(args []string, stderr io.Writer) (options, error) {
	fs := flag.NewFlagSet("llmcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		o              options
		tasks, checks  string
		allChecksUsage = strings.Join(allChecks(), ", ")
	)

	fs.StringVar(&o.config, "config", "", "файл маршрутов llmconfig; секреты — из окружения")
	fs.StringVar(&tasks, "tasks", "", "задачи через запятую; пусто — все объявленные")
	fs.StringVar(&checks, "checks", strings.Join(defaultChecks(), ","),
		"проверки через запятую из "+allChecksUsage+"; limits шлёт предельное число кадров и по умолчанию выключена")
	fs.IntVar(&o.maxRequests, "max-requests", defaultMaxRequests,
		"потолок обращений: попытки вызова плеч, включая повторы клиента, и запросы проверки oauth; "+
			"внутренние запросы адаптера (токен, загрузка и удаление файлов) не считаются")
	fs.Int64Var(&o.maxCost, "max-cost", defaultMaxCost,
		"порог оценённого расхода в микроединицах валюты: проверка за ним не начинается, а вызов, "+
			"перелетевший его, оплачен целиком и даёт выход 3; расход без тарифа держит только -max-requests")
	fs.DurationVar(&o.timeout, "timeout", defaultTimeout, "срок прогона целиком")

	if err := fs.Parse(args); err != nil {
		return options{}, err
	}

	o.tasks = splitList(tasks)
	o.checks = splitList(checks)

	var errs []error

	if fs.NArg() > 0 {
		errs = append(errs, fmt.Errorf("лишние аргументы: %s", strings.Join(fs.Args(), " ")))
	}

	if o.config == "" {
		errs = append(errs, errors.New("-config: файл маршрутов не задан"))
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

	if len(o.checks) == 0 {
		errs = append(errs, errors.New("-checks: проверки не выбраны"))
	}

	for _, c := range o.checks {
		if !slices.Contains(allChecks(), c) {
			errs = append(errs, fmt.Errorf("-checks: неизвестная проверка %q", c))
		}
	}

	return o, errors.Join(errs...)
}

// splitList — элементы списка через запятую без пустых и повторов, в исходном порядке.
func splitList(s string) []string {
	var out []string

	for item := range strings.SplitSeq(s, ",") {
		item = strings.TrimSpace(item)
		if item != "" && !slices.Contains(out, item) {
			out = append(out, item)
		}
	}

	return out
}

func run(ctx context.Context, o options, stdout, stderr io.Writer, lookup llmconfig.Lookup) int {
	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()

	data, err := os.ReadFile(o.config)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "llmcheck:", err)

		return exitUsage
	}

	r, err := prepare(data, o, lookup)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "llmcheck:", err)

		return exitUsage
	}

	r.out = stdout
	r.header(data, o)
	r.runAll(ctx, o.checks)

	return r.footer()
}

// prepare разбирает конфиг, оставляет выбранные задачи и собирает их плечи за общим счётчиком.
func prepare(data []byte, o options, lookup llmconfig.Lookup) (*runner, error) {
	cfg, err := llmconfig.Parse(data)
	if err != nil {
		return nil, err
	}

	selected := o.tasks
	if len(selected) == 0 {
		selected = slices.Sorted(maps.Keys(cfg.Tasks))
	}

	for _, name := range selected {
		if _, ok := cfg.Tasks[name]; !ok {
			return nil, fmt.Errorf("-tasks: задача %q не объявлена", name)
		}
	}

	maps.DeleteFunc(cfg.Tasks, func(name string, _ llmconfig.Task) bool { return !slices.Contains(selected, name) })

	if err = cfg.Expand(lookup); err != nil {
		return nil, err
	}

	routes, err := cfg.Routes()
	if err != nil {
		return nil, err
	}

	r := &runner{
		cfg:     cfg,
		tasks:   selected,
		meter:   &meter{limit: o.maxRequests},
		maxCost: o.maxCost,
		spent:   map[string]int64{},
		seen:    map[string]string{},
		oauth:   map[string]oauthTarget{},
		router:  &llm.Router{Providers: map[string]llm.Provider{}, Tasks: routes},
	}

	for _, name := range cfg.Active() {
		p, target, buildErr := build(cfg.Providers[name])
		if buildErr != nil {
			return nil, fmt.Errorf("providers.%s: %w", name, buildErr)
		}

		r.router.Providers[name] = counted{Provider: p, meter: r.meter}

		if target != nil {
			r.oauth[name] = *target
		}
	}

	if err = r.router.Validate(nil); err != nil {
		return nil, err
	}

	return r, nil
}

// oauthTarget — всё, что нужно запросу токена GigaChat мимо адаптера: срок токена адаптер наружу не отдаёт.
type oauthTarget struct {
	endpoint string
	key      string
	scope    string
	http     *http.Client
}

// build собирает плечо по виду; у GigaChat — ещё и цель проверки OAuth тем же транспортом.
func build(p llmconfig.Provider) (llm.Provider, *oauthTarget, error) {
	pool, err := loadCA(p.CAFile)
	if err != nil {
		return nil, nil, err
	}

	client := trusting(pool)

	switch p.Kind {
	case llmconfig.KindOllama:
		prov := ollama.New(p.Endpoint)
		prov.HTTP = client

		return prov, nil, nil
	case llmconfig.KindGigaChat:
		prov, buildErr := gigachat.New(gigachat.Config{
			OAuthEndpoint:    p.OAuthEndpoint,
			APIEndpoint:      p.Endpoint,
			AuthorizationKey: p.Auth.AuthorizationKey.Value(),
			Scope:            p.Scope,
			CA:               pool,
			HTTP:             client,
		})
		target := &oauthTarget{
			endpoint: p.OAuthEndpoint,
			key:      p.Auth.AuthorizationKey.Value(),
			scope:    p.Scope,
			http:     client,
		}

		return prov, target, buildErr
	case llmconfig.KindYandex:
		prov, buildErr := yandex.New(yandex.Config{
			Endpoint:    p.Endpoint,
			Folder:      p.Folder,
			Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
			HTTP:        client,
		})

		return prov, nil, buildErr
	default:
		return nil, nil, fmt.Errorf("неизвестный вид плеча %q", p.Kind)
	}
}

// loadCA — корни из PEM-файла целиком, без системных; пустой путь — корни системы.
func loadCA(path string) (*x509.CertPool, error) {
	if path == "" {
		return nil, nil //nolint:nilnil // пустой пул — корни системы, это не отказ
	}

	pem, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ca_file: %w", err)
	}

	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("ca_file: в %s нет сертификатов PEM", path)
	}

	return pool, nil
}

func trusting(pool *x509.CertPool) *http.Client {
	if pool == nil {
		return &http.Client{}
	}

	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return &http.Client{}
	}

	t := base.Clone()
	t.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}

	return &http.Client{Transport: t}
}

// header — шапка протокола: дата, сборка, отпечаток конфига, плечи и задачи без секретов.
func (r *runner) header(data []byte, o options) {
	sum := sha256.Sum256(data)

	r.printf("# llmcheck %s\n\n", time.Now().Format(time.DateOnly))
	r.printf("- сборка: %s\n", buildLine())
	r.printf("- конфиг: sha256 %s\n", hex.EncodeToString(sum[:configHashBytes]))
	r.printf("- потолки: обращений %d, расхода %d мк. на валюту\n", o.maxRequests, o.maxCost)
	r.printf("- проверки: %s\n\n## Плечи\n\n", strings.Join(o.checks, ", "))

	for _, name := range r.cfg.Active() {
		r.printf("- %s\n", providerLine(name, r.cfg.Providers[name]))
	}

	r.printf("\n## Задачи\n\n")

	for _, name := range r.tasks {
		r.printf("- %s\n", r.taskLine(name))
	}

	r.printf("\n## Проверки\n\n")
}

func providerLine(name string, p llmconfig.Provider) string {
	parts := []string{name + ": " + p.Kind, "endpoint " + p.Endpoint}

	if p.OAuthEndpoint != "" {
		parts = append(parts, "oauth "+p.OAuthEndpoint, "scope "+p.Scope)
	}

	if p.Folder != "" {
		parts = append(parts, "каталог задан")
	}

	if p.CAFile != "" {
		parts = append(parts, "свои корни TLS")
	}

	return strings.Join(parts, ", ")
}

func (r *runner) taskLine(name string) string {
	task := r.cfg.Tasks[name]
	desc := r.route(name).Descriptor()

	parts := []string{
		fmt.Sprintf("%s: %s/%s", name, task.Provider, desc.Model),
		"ревизия " + orDash(desc.Revision),
		"ответ " + string(desc.Output.Mode),
	}

	if desc.Output.Name != "" || desc.Output.Strict {
		parts = append(parts, fmt.Sprintf("схема %s strict=%t", orDash(desc.Output.Name), desc.Output.Strict))
	}

	if desc.Options.Reasoning != "" {
		parts = append(parts, "reasoning "+string(desc.Options.Reasoning))
	}

	if desc.Options.Temperature != nil {
		parts = append(parts, fmt.Sprintf("температура %g", *desc.Options.Temperature))
	}

	if desc.Options.MaxOutputTokens > 0 {
		parts = append(parts, fmt.Sprintf("max_output_tokens %d", desc.Options.MaxOutputTokens))
	}

	parts = append(parts, "бюджет "+r.route(name).Budget().String())

	if task.PricePlan != "" {
		plans, _ := r.cfg.PricePlans(task.PricePlan)

		revisions := make([]string, 0, len(plans))
		for _, p := range plans {
			revisions = append(revisions, p.Revision)
		}

		parts = append(parts, fmt.Sprintf("тариф %s (%s)", task.PricePlan, strings.Join(revisions, ", ")))
	}

	return strings.Join(parts, ", ")
}

// footer — итог и код выхода.
func (r *runner) footer() int {
	counts := map[status]int{}
	for _, res := range r.results {
		counts[res.status]++
	}

	spent := make([]string, 0, len(r.spent))
	for _, currency := range slices.Sorted(maps.Keys(r.spent)) {
		spent = append(spent, fmt.Sprintf("%s %d мк.", currency, r.spent[currency]))
	}

	r.printf("\n## Итог\n\n- пройдено %d, нарушений %d, пропущено %d\n",
		counts[statusPass], counts[statusFail], counts[statusSkip])
	r.printf(
		"- обращений %d из %d; оценённый расход: %s\n",
		r.meter.used,
		r.meter.limit,
		orDash(strings.Join(spent, ", ")),
	)

	if r.overspent() {
		r.incomplete = true
		r.printf("- потолок расхода %d мк. превышен\n", r.maxCost)
	}

	switch {
	case counts[statusFail] > 0:
		return exitViolation
	case r.incomplete:
		r.printf("- прогон неполный: остановлен потолком или сроком\n")

		return exitIncomplete
	default:
		return exitOK
	}
}

// buildLine — версия модуля и ревизия VCS сборки.
func buildLine() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "неизвестна"
	}

	line := info.Main.Path + " " + info.Main.Version

	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			line += " " + s.Value
		case "vcs.modified":
			if s.Value == "true" {
				line += " с незакоммиченными правками"
			}
		}
	}

	return line
}

func orDash(s string) string {
	if s == "" {
		return "—"
	}

	return s
}
