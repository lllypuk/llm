// Package llmconfig — маршруты задач в JSON: плечи, задачи, тарифы. Разбор строгий, ошибки копятся
// с путём; секреты задаются ссылками `${ИМЯ}` и разворачиваются только у плеч, на которые ссылается
// задача. Собирать плечи и транспорты — дело потребителя.
package llmconfig

import (
	"errors"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/pricing"
)

// Виды плеч.
const (
	KindOllama   = "ollama"
	KindGigaChat = "gigachat"
	KindYandex   = "yandex"
)

// Config — файл маршрутов целиком. Повреждённый не заменяется ничем: запасного конфига нет.
type Config struct {
	Providers map[string]Provider `json:"providers"`
	Tasks     map[string]Task     `json:"tasks"`
	Prices    map[string][]Price  `json:"prices,omitempty"`
}

// Provider — плечо: адреса, подпись и корни TLS. Поля, чужие виду плеча, отбиваются.
type Provider struct {
	Kind          string `json:"kind"`
	Endpoint      string `json:"endpoint"`
	OAuthEndpoint string `json:"oauth_endpoint,omitempty"`
	Scope         string `json:"scope,omitempty"`
	Folder        string `json:"folder,omitempty"`
	Auth          Auth   `json:"auth,omitzero"`
	CAFile        string `json:"ca_file,omitempty"`
}

// Auth — секреты плеча: ключ авторизации GigaChat или ключ API Яндекса.
type Auth struct {
	AuthorizationKey Secret `json:"authorization_key,omitzero"`
	APIKey           Secret `json:"api_key,omitzero"`
}

// Task — привязка задачи; срок попытки — строкой длительности Go (`90s`).
type Task struct {
	Provider        string   `json:"provider"`
	Model           string   `json:"model"`
	Revision        string   `json:"revision,omitempty"`
	Output          Output   `json:"output,omitzero"`
	Reasoning       string   `json:"reasoning,omitempty"`
	MaxOutputTokens int      `json:"max_output_tokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	AttemptTimeout  string   `json:"attempt_timeout,omitempty"`
	Attempts        int      `json:"attempts,omitempty"`
	PricePlan       string   `json:"price_plan,omitempty"`
}

// Output — форма ответа без схемы: схема — артефакт потребителя.
type Output struct {
	Mode       string `json:"mode,omitempty"`
	SchemaName string `json:"schema_name,omitempty"`
	Strict     bool   `json:"strict,omitempty"`
}

// Price — ревизия тарифа; начало действия — RFC 3339.
type Price struct {
	Revision  string `json:"revision"`
	Currency  string `json:"currency,omitempty"`
	ValidFrom string `json:"valid_from"`
	Free      bool   `json:"free,omitempty"`
	Rates     Rates  `json:"rates,omitzero"`
}

// Rates — микроединицы валюты за миллион токенов; отсутствующее поле — тарифа нет.
type Rates struct {
	BillableInput *int64 `json:"billable_input,omitempty"`
	CachedInput   *int64 `json:"cached_input,omitempty"`
	Reasoning     *int64 `json:"reasoning,omitempty"`
	Output        *int64 `json:"output,omitempty"`
}

// Lookup — источник переменных окружения; os.LookupEnv подходит как есть.
type Lookup func(name string) (string, bool)

// Load разбирает, проверяет и разворачивает секреты; ошибки проверки и секретов копятся вместе.
func Load(data []byte, lookup Lookup) (*Config, error) {
	c, err := decode(data)
	if err != nil {
		return nil, err
	}

	if err = errors.Join(c.Validate(), c.Expand(lookup)); err != nil {
		return nil, err
	}

	return c, nil
}

// Parse разбирает и проверяет файл целиком, не читая секретов: потребитель, сужающий набор задач,
// зовёт Expand после сужения, и ключи плеч невыбранных задач ему не нужны.
func Parse(data []byte) (*Config, error) {
	c, err := decode(data)
	if err != nil {
		return nil, err
	}

	if err = c.Validate(); err != nil {
		return nil, err
	}

	return c, nil
}

func decode(data []byte) (*Config, error) {
	var c Config
	if err := strict(data, &c); err != nil {
		return nil, err
	}

	return &c, nil
}

// Validate проверяет связность и значения без окружения и сети.
func (c *Config) Validate() error {
	var errs []error

	for _, name := range slices.Sorted(maps.Keys(c.Providers)) {
		errs = append(errs, c.Providers[name].validate("providers."+name)...)
	}

	for _, name := range slices.Sorted(maps.Keys(c.Tasks)) {
		path := "tasks." + name
		task := c.Tasks[name]

		if _, ok := c.Providers[task.Provider]; !ok {
			errs = append(errs, fmt.Errorf("%s.provider: плечо %q не объявлено", path, task.Provider))
		}

		if _, ok := c.Prices[task.PricePlan]; task.PricePlan != "" && !ok {
			errs = append(errs, fmt.Errorf("%s.price_plan: тариф %q не объявлен", path, task.PricePlan))
		}

		_, taskErrs := task.route(path)
		errs = append(errs, taskErrs...)
	}

	for _, name := range slices.Sorted(maps.Keys(c.Prices)) {
		_, priceErrs := plans("prices."+name, c.Prices[name])
		errs = append(errs, priceErrs...)
	}

	return errors.Join(errs...)
}

// Active — плечи, на которые ссылается хотя бы одна задача, по имени.
func (c *Config) Active() []string {
	var names []string

	for _, task := range c.Tasks {
		if _, ok := c.Providers[task.Provider]; ok && !slices.Contains(names, task.Provider) {
			names = append(names, task.Provider)
		}
	}

	slices.Sort(names)

	return names
}

// Expand разворачивает секреты активных плеч после Validate: переменная не задана или пуста — ошибка
// с путём и именем переменной, без значения. Секреты неактивных плеч не читаются.
func (c *Config) Expand(lookup Lookup) error {
	var errs []error

	for _, name := range c.Active() {
		p := c.Providers[name]
		for _, s := range []struct {
			key    string
			secret *Secret
		}{
			{gigachatAuthField, &p.Auth.AuthorizationKey},
			{yandexAuthField, &p.Auth.APIKey},
		} {
			if err := s.secret.expand(lookup); err != nil {
				errs = append(errs, fmt.Errorf("providers.%s.%s: %w", name, s.key, err))
			}
		}

		c.Providers[name] = p
	}

	return errors.Join(errs...)
}

// Routes — задачи маршрутизатора; ошибка — конфиг не прошёл бы Validate.
func (c *Config) Routes() (map[string]llm.TaskConfig, error) {
	routes := make(map[string]llm.TaskConfig, len(c.Tasks))

	var errs []error

	for name, task := range c.Tasks {
		route, taskErrs := task.route("tasks." + name)
		routes[name] = route
		errs = append(errs, taskErrs...)
	}

	return routes, errors.Join(errs...)
}

// PricePlans — ревизии тарифа по имени; неизвестное имя — ошибка.
func (c *Config) PricePlans(name string) ([]pricing.PricePlan, error) {
	prices, ok := c.Prices[name]
	if !ok {
		return nil, fmt.Errorf("prices.%s: тариф не объявлен", name)
	}

	out, errs := plans("prices."+name, prices)

	return out, errors.Join(errs...)
}

// badAddress — почему адрес плеча негоден; сам адрес в ответ не попадает: в нём мог оказаться секрет.
// Учётные данные, query и фрагмент отбиваются — адрес печатается в протоколы и логи как есть.
func badAddress(addr string) string {
	u, err := url.Parse(addr)

	switch {
	case err != nil:
		return "не URL"
	case u.Scheme != "http" && u.Scheme != "https":
		return "схема не http и не https"
	case u.Host == "":
		return "хост не задан"
	case u.User != nil:
		return "учётные данные в адресе запрещены: секрет задаётся полем auth"
	case u.RawQuery != "" || u.ForceQuery || u.Fragment != "":
		return "query и фрагмент в адресе запрещены"
	default:
		return ""
	}
}

// Поля секретов в путях ошибок.
const (
	gigachatAuthField  = "auth.authorization_key"
	oauthEndpointField = "oauth_endpoint"
	yandexAuthField    = "auth.api_key"
)

// field — поле плеча для сверки с видом: задано ли и обязательно ли.
type field struct {
	name string
	set  bool
}

func (p Provider) validate(path string) []error {
	var errs []error

	fields := []field{
		{oauthEndpointField, p.OAuthEndpoint != ""},
		{"scope", p.Scope != ""},
		{"folder", p.Folder != ""},
		{gigachatAuthField, !p.Auth.AuthorizationKey.IsZero()},
		{yandexAuthField, !p.Auth.APIKey.IsZero()},
	}

	var required []string

	switch p.Kind {
	case KindOllama:
	case KindGigaChat:
		required = []string{oauthEndpointField, "scope", gigachatAuthField}
	case KindYandex:
		required = []string{"folder", yandexAuthField}
	default:
		return append(errs, fmt.Errorf("%s.kind: неизвестный вид плеча %q", path, p.Kind))
	}

	if p.Endpoint == "" {
		errs = append(errs, fmt.Errorf("%s.endpoint: адрес не задан", path))
	}

	for key, addr := range map[string]string{"endpoint": p.Endpoint, oauthEndpointField: p.OAuthEndpoint} {
		if msg := badAddress(addr); addr != "" && msg != "" {
			errs = append(errs, fmt.Errorf("%s.%s: %s", path, key, msg))
		}
	}

	for _, f := range fields {
		switch need := slices.Contains(required, f.name); {
		case need && !f.set:
			errs = append(errs, fmt.Errorf("%s.%s: обязательно у плеча %s", path, f.name, p.Kind))
		case !need && f.set:
			errs = append(errs, fmt.Errorf("%s.%s: не относится к плечу %s", path, f.name, p.Kind))
		}
	}

	for _, s := range []struct {
		key    string
		secret Secret
	}{
		{gigachatAuthField, p.Auth.AuthorizationKey},
		{yandexAuthField, p.Auth.APIKey},
	} {
		if _, err := s.secret.name(); !s.secret.IsZero() && err != nil {
			errs = append(errs, fmt.Errorf("%s.%s: %w", path, s.key, err))
		}
	}

	return errs
}

func (t Task) route(path string) (llm.TaskConfig, []error) {
	route := llm.TaskConfig{
		Provider: t.Provider,
		Model:    t.Model,
		Revision: t.Revision,
		Output:   llm.Output{Mode: llm.Mode(t.Output.Mode), Name: t.Output.SchemaName, Strict: t.Output.Strict},
		Options: llm.Options{
			Temperature:     t.Temperature,
			Reasoning:       llm.Effort(t.Reasoning),
			MaxOutputTokens: t.MaxOutputTokens,
		},
		Attempts:  t.Attempts,
		PricePlan: t.PricePlan,
	}

	var errs []error

	probe := llm.Request{Model: route.Model, Output: route.Output, Options: route.Options}
	if probe.Output.Mode == llm.ModeSchema {
		probe.Output.Schema = []byte(`{}`)
	}

	if err := probe.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("%s: %w", path, err))
	}

	if t.Attempts < 0 {
		errs = append(errs, fmt.Errorf("%s.attempts: отрицательное число попыток", path))
	}

	if t.AttemptTimeout != "" {
		d, err := time.ParseDuration(t.AttemptTimeout)

		switch {
		case err != nil:
			errs = append(errs, fmt.Errorf("%s.attempt_timeout: не длительность: %q", path, t.AttemptTimeout))
		case d <= 0:
			errs = append(errs, fmt.Errorf("%s.attempt_timeout: срок не положителен", path))
		default:
			route.AttemptTimeout = d
		}
	}

	return route, errs
}

// plans — ревизии одного тарифа: у каждой свои начало действия и имя, иначе попытке достаётся
// ревизия порядком перечисления, а не временем.
func plans(path string, prices []Price) ([]pricing.PricePlan, []error) {
	if len(prices) == 0 {
		return nil, []error{fmt.Errorf("%s: ревизий нет", path)}
	}

	var (
		out  []pricing.PricePlan
		errs []error
	)

	for i, price := range prices {
		at := fmt.Sprintf("%s[%d]", path, i)

		from, err := time.Parse(time.RFC3339, price.ValidFrom)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s.valid_from: не время RFC 3339: %q", at, price.ValidFrom))
		}

		plan := pricing.PricePlan{
			Revision:  price.Revision,
			Currency:  price.Currency,
			ValidFrom: from,
			Free:      price.Free,
			Rates: pricing.Rates{
				BillableInput: price.Rates.BillableInput,
				CachedInput:   price.Rates.CachedInput,
				Reasoning:     price.Rates.Reasoning,
				Output:        price.Rates.Output,
			},
		}

		if planErr := plan.Validate(); planErr != nil {
			errs = append(errs, fmt.Errorf("%s: %w", at, planErr))
		}

		for _, prev := range out {
			if err == nil && prev.ValidFrom.Equal(from) {
				errs = append(errs, fmt.Errorf("%s.valid_from: начало совпадает с ревизией %s", at, prev.Revision))
			}

			if prev.Revision == plan.Revision {
				errs = append(errs, fmt.Errorf("%s.revision: ревизия %s повторена", at, plan.Revision))
			}
		}

		out = append(out, plan)
	}

	return out, errs
}

// Secret — секрет плеча: в файле только ссылка `${ИМЯ}`, значение — после Expand. Печатается ссылкой.
type Secret struct { //nolint:recvcheck // UnmarshalText меняет значение, а String обязан работать и у неадресуемой копии
	ref   string
	value string
}

// UnmarshalText запоминает ссылку как есть; синтаксис сверяет Validate.
func (s *Secret) UnmarshalText(text []byte) error {
	s.ref = string(text)

	return nil
}

// MarshalText — ссылка, не значение.
func (s Secret) MarshalText() ([]byte, error) { return []byte(s.ref), nil }

// String — ссылка, не значение: секрет не уезжает ни в лог, ни в ошибку.
func (s Secret) String() string { return s.ref }

// GoString — [Secret.String] и для `%#v`.
func (s Secret) GoString() string { return s.ref }

// IsZero — ссылки в файле нет.
func (s Secret) IsZero() bool { return s.ref == "" }

// Value — развёрнутое значение; пустое до Expand.
func (s Secret) Value() string { return s.value }

// name — имя переменной из ссылки. Буквальное значение отбивается, не повторяясь в ошибке.
func (s Secret) name() (string, error) {
	name, ok := strings.CutPrefix(s.ref, "${")
	if ok {
		name, ok = strings.CutSuffix(name, "}")
	}

	if !ok || !envName(name) {
		return "", errors.New("секрет задаётся ссылкой ${ИМЯ}, а не значением")
	}

	return name, nil
}

func (s *Secret) expand(lookup Lookup) error {
	if s.IsZero() {
		return nil
	}

	name, err := s.name()
	if err != nil {
		return nil //nolint:nilerr // синтаксис ссылки отбивает Validate, второй раз та же ошибка не нужна
	}

	value, ok := "", false
	if lookup != nil {
		value, ok = lookup(name)
	}

	switch {
	case !ok:
		return fmt.Errorf("переменная %s не задана", name)
	case value == "":
		return fmt.Errorf("переменная %s пуста", name)
	}

	s.value = value

	return nil
}

func envName(name string) bool {
	if name == "" || (name[0] >= '0' && name[0] <= '9') {
		return false
	}

	for _, r := range name {
		if r != '_' && (r < 'A' || r > 'Z') && (r < 'a' || r > 'z') && (r < '0' || r > '9') {
			return false
		}
	}

	return true
}
