package llmconfig_test

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/pricing"
)

const devConfig = `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "http://ollama:11434"},
    "giga": {
      "kind": "gigachat",
      "endpoint": "https://api.giga.chat/api/v1",
      "oauth_endpoint": "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
      "scope": "GIGACHAT_API_PERS",
      "auth": {"authorization_key": "${GIGACHAT_AUTH_KEY}"},
      "ca_file": "/etc/llm/russian_trusted_root_ca.pem"
    },
    "ya": {
      "kind": "yandex",
      "endpoint": "https://ai.api.cloud.yandex.net/v1",
      "folder": "b1g",
      "auth": {"api_key": "${YANDEX_API_KEY}"}
    }
  },
  "tasks": {
    "nameplate": {
      "provider": "local",
      "model": "gemma",
      "output": {"mode": "json"},
      "attempt_timeout": "90s",
      "attempts": 3,
      "price_plan": "free"
    },
    "ask": {
      "provider": "local",
      "model": "gemma",
      "revision": "r1",
      "output": {"mode": "schema", "schema_name": "answer", "strict": true},
      "reasoning": "low",
      "max_output_tokens": 800,
      "temperature": 0.1
    }
  },
  "prices": {
    "free": [{"revision": "ollama", "valid_from": "2026-01-01T00:00:00+03:00", "free": true}],
    "giga-pro": [
      {"revision": "2026-09", "currency": "RUB", "valid_from": "2026-09-01T00:00:00+03:00",
       "rates": {"billable_input": 1500000000, "cached_input": 0, "output": 1500000000}}
    ]
  }
}`

// lookup — окружение из карты, запоминающее спрошенные имена.
type lookup struct {
	env   map[string]string
	asked []string
}

func (l *lookup) get(name string) (string, bool) {
	l.asked = append(l.asked, name)
	v, ok := l.env[name]

	return v, ok
}

// TestLoadDevConfigNeedsNoCloudKeys — задачи на Ollama: облачные плечи объявлены, но их ключи
// из окружения не читаются и не требуются; маршруты и тарифы собираются с длительностями.
func TestLoadDevConfigNeedsNoCloudKeys(t *testing.T) {
	t.Parallel()

	env := &lookup{}

	cfg, err := llmconfig.Load([]byte(devConfig), env.get)
	if err != nil {
		t.Fatal(err)
	}

	if len(env.asked) != 0 {
		t.Fatalf("спрошены переменные неактивных плеч: %v", env.asked)
	}

	if got := cfg.Active(); !slices.Equal(got, []string{"local"}) {
		t.Fatalf("активные плечи %v", got)
	}

	routes, err := cfg.Routes()
	if err != nil {
		t.Fatal(err)
	}

	want := llm.TaskConfig{
		Provider:       "local",
		Model:          "gemma",
		Output:         llm.Output{Mode: llm.ModeJSON},
		AttemptTimeout: 90 * time.Second,
		Attempts:       3,
		PricePlan:      "free",
	}
	if fmt.Sprint(routes["nameplate"]) != fmt.Sprint(want) {
		t.Fatalf("nameplate %+v", routes["nameplate"])
	}

	ask := routes["ask"]
	if ask.Output.Mode != llm.ModeSchema || ask.Output.Name != "answer" || !ask.Output.Strict || ask.Revision != "r1" ||
		ask.Options.Reasoning != llm.EffortLow || ask.Options.MaxOutputTokens != 800 ||
		ask.Options.Temperature == nil || *ask.Options.Temperature != 0.1 || ask.AttemptTimeout != 0 {
		t.Fatalf("ask %+v", ask)
	}

	plans, err := cfg.PricePlans("giga-pro")
	if err != nil {
		t.Fatal(err)
	}

	if len(plans) != 1 || plans[0].Currency != "RUB" || *plans[0].Rates.BillableInput != 1_500_000_000 ||
		*plans[0].Rates.CachedInput != 0 || plans[0].Rates.Reasoning != nil ||
		!plans[0].ValidFrom.Equal(time.Date(2026, 8, 31, 21, 0, 0, 0, time.UTC)) {
		t.Fatalf("тариф %+v", plans)
	}

	if est := pricing.Estimate(llm.AttemptReport{StartedAt: plans[0].ValidFrom}, plans[0]); est.Status == "" {
		t.Fatal("тариф из конфига не годен оценке")
	}

	if _, missingErr := cfg.PricePlans("missing"); missingErr == nil {
		t.Fatal("неизвестный тариф выдан")
	}
}

// TestLoadExpandsOnlyActiveSecrets — задача на GigaChat требует его ключ и только его; значение секрета
// не печатается ни конфигом, ни ошибкой.
func TestLoadExpandsOnlyActiveSecrets(t *testing.T) {
	t.Parallel()

	data := strings.Replace(devConfig, `"provider": "local",
      "model": "gemma",
      "revision"`, `"provider": "giga",
      "model": "GigaChat-2-Max",
      "revision"`, 1)

	const key = "s3cr3t-authorization-key"

	env := &lookup{env: map[string]string{"GIGACHAT_AUTH_KEY": key, "YANDEX_API_KEY": "yandex-secret"}}

	cfg, err := llmconfig.Load([]byte(data), env.get)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(env.asked, []string{"GIGACHAT_AUTH_KEY"}) {
		t.Fatalf("спрошены %v", env.asked)
	}

	giga := cfg.Providers["giga"]
	if giga.Auth.AuthorizationKey.Value() != key || cfg.Providers["ya"].Auth.APIKey.Value() != "" {
		t.Fatal("секреты развёрнуты не у тех плеч")
	}

	for _, format := range []string{"%v", "%+v", "%#v", "%s"} {
		if printed := fmt.Sprintf(format, cfg); strings.Contains(printed, key) {
			t.Fatalf("%s печатает секрет: %s", format, printed)
		}
	}

	_, err = llmconfig.Load([]byte(data), (&lookup{env: map[string]string{"GIGACHAT_AUTH_KEY": ""}}).get)
	if err == nil ||
		!strings.Contains(err.Error(), "providers.giga.auth.authorization_key: переменная GIGACHAT_AUTH_KEY пуста") {
		t.Fatalf("пустая переменная: %v", err)
	}

	_, err = llmconfig.Load([]byte(data), nil)
	if err == nil || !strings.Contains(err.Error(), "переменная GIGACHAT_AUTH_KEY не задана") {
		t.Fatalf("незаданная переменная: %v", err)
	}

	literal := strings.Replace(data, "${GIGACHAT_AUTH_KEY}", key, 1)

	_, err = llmconfig.Load([]byte(literal), env.get)
	if err == nil || !strings.Contains(err.Error(), "providers.giga.auth.authorization_key: секрет задаётся ссылкой") {
		t.Fatalf("буквальный секрет: %v", err)
	}

	if strings.Contains(err.Error(), key) {
		t.Fatalf("ошибка печатает секрет: %v", err)
	}
}

// TestParseLeavesSecretsForNarrowedTasks — Parse секретов не читает, и Expand после сужения задач
// не спрашивает ключ плеча, на которое ссылалась только отброшенная задача.
func TestParseLeavesSecretsForNarrowedTasks(t *testing.T) {
	t.Parallel()

	data := strings.Replace(devConfig, `"provider": "local",
      "model": "gemma",
      "revision"`, `"provider": "giga",
      "model": "GigaChat-2-Max",
      "revision"`, 1)

	cfg, err := llmconfig.Parse([]byte(data))
	if err != nil {
		t.Fatal(err)
	}

	env := &lookup{env: map[string]string{}}

	maps.DeleteFunc(cfg.Tasks, func(_ string, task llmconfig.Task) bool { return task.Provider == "giga" })

	if err = cfg.Expand(env.get); err != nil || len(env.asked) != 0 {
		t.Fatalf("Expand: %v, спрошены %v", err, env.asked)
	}

	if _, err = llmconfig.Parse([]byte(`{"providers": {"o": {"kind": "ollama"}}}`)); err == nil {
		t.Fatal("Parse не проверил файл")
	}
}

// TestLoadIsStrict — неизвестные поля, повтор ключа и чужие типы отбиваются все сразу, с путём.
func TestLoadIsStrict(t *testing.T) {
	t.Parallel()

	data := `{
	  "providers": {"local": {"kind": "ollama", "endpoint": "http://o", "auth": {"token": "x"}}},
	  "tasks": {
	    "ask": {"provider": "local", "model": "m", "atempts": 3, "attempts": "3"},
	    "nameplate": {"provider": "local", "model": "m"},
	    "nameplate": {"provider": "local", "model": "m", "temperature": "hot"}
	  },
	  "prices": {"free": {"revision": "x"}},
	  "extra": true
	}`

	_, err := llmconfig.Load([]byte(data), nil)
	if err == nil {
		t.Fatal("ожидался отказ")
	}

	for _, want := range []string{
		"providers.local.auth.token: неизвестное поле",
		"tasks.ask.atempts: неизвестное поле",
		"tasks.ask.attempts: ожидается целое число",
		"tasks.nameplate.temperature: ожидается число",
		"tasks.nameplate: ключ повторён",
		"prices.free: ожидается список",
		"extra: неизвестное поле",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("нет %q в\n%v", want, err)
		}
	}

	for name, doc := range map[string]string{
		"второй документ": `{"providers": {}, "tasks": {}} {}`,
		"обрыв":           `{"providers": {`,
		"не объект":       `[]`,
		"null поля":       `{"providers": null, "tasks": {}}`,
	} {
		if _, docErr := llmconfig.Load([]byte(doc), nil); docErr == nil {
			t.Errorf("%s: принят", name)
		}
	}
}

// TestValidateCollects — связность, поля вида плеча, формы, длительности и тарифы — все ошибки разом.
func TestValidateCollects(t *testing.T) {
	t.Parallel()

	data := `{
	  "providers": {
	    "local": {"kind": "ollama", "endpoint": "http://o", "scope": "x", "auth": {"api_key": "${K}"}},
	    "giga": {"kind": "gigachat", "endpoint": "https://g"},
	    "ya": {"kind": "yandex", "endpoint": "", "folder": "f", "auth": {"api_key": "$K"}},
	    "odd": {"kind": "openai", "endpoint": "https://o"},
	    "leak": {"kind": "ollama", "endpoint": "http://user:hunter2@o"},
	    "query": {"kind": "ollama", "endpoint": "http://o/?key=hunter2"},
	    "bare": {"kind": "ollama", "endpoint": "ollama:11434"}
	  },
	  "tasks": {
	    "a": {"provider": "nowhere", "model": "m", "price_plan": "nope"},
	    "b": {"provider": "local", "model": "", "output": {"mode": "xml"}},
	    "c": {"provider": "local", "model": "m", "output": {"mode": "json", "schema_name": "n"}},
	    "d": {"provider": "local", "model": "m", "attempt_timeout": "soon", "attempts": -1},
	    "e": {"provider": "local", "model": "m", "attempt_timeout": "-1s", "reasoning": "max"}
	  },
	  "prices": {
	    "empty": [],
	    "dup": [
	      {"revision": "r1", "currency": "RUB", "valid_from": "2026-09-01T00:00:00Z"},
	      {"revision": "r1", "currency": "RUB", "valid_from": "2026-09-01T00:00:00Z"},
	      {"revision": "r,2", "currency": "", "valid_from": "1 сентября"}
	    ]
	  }
	}`

	_, err := llmconfig.Load([]byte(data), nil)
	if err == nil {
		t.Fatal("ожидался отказ")
	}

	for _, want := range []string{
		"providers.local.scope: не относится к плечу ollama",
		"providers.local.auth.api_key: не относится к плечу ollama",
		"providers.giga.oauth_endpoint: обязательно у плеча gigachat",
		"providers.giga.scope: обязательно у плеча gigachat",
		"providers.giga.auth.authorization_key: обязательно у плеча gigachat",
		"providers.ya.endpoint: адрес не задан",
		"providers.ya.auth.api_key: секрет задаётся ссылкой",
		`providers.odd.kind: неизвестный вид плеча "openai"`,
		"providers.leak.endpoint: учётные данные в адресе запрещены",
		"providers.query.endpoint: query и фрагмент в адресе запрещены",
		"providers.bare.endpoint: схема не http и не https",
		`tasks.a.provider: плечо "nowhere" не объявлено`,
		`tasks.a.price_plan: тариф "nope" не объявлен`,
		"tasks.b: модель не задана",
		"tasks.c: имя схемы и strict без режима schema",
		`tasks.d.attempt_timeout: не длительность: "soon"`,
		"tasks.d.attempts: отрицательное число попыток",
		"tasks.e.attempt_timeout: срок не положителен",
		"tasks.e: неизвестная глубина рассуждений max",
		"prices.empty: ревизий нет",
		"prices.dup[1].valid_from: начало совпадает с ревизией r1",
		"prices.dup[1].revision: ревизия r1 повторена",
		`prices.dup[2].valid_from: не время RFC 3339: "1 сентября"`,
		"prices.dup[2]: pricing: r,2: запятая в ревизии",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("нет %q в\n%v", want, err)
		}
	}

	if strings.Contains(err.Error(), "hunter2") {
		t.Errorf("адрес с секретом попал в ошибку: %v", err)
	}

	if !strings.Contains(err.Error(), "tasks.b: неизвестный режим ответа xml") &&
		!strings.Contains(err.Error(), "tasks.b: модель не задана") {
		t.Errorf("форма xml не отбита: %v", err)
	}
}

// TestRoutesFeedRouter — маршруты конфига разрешаются роутером с профилем плеча без правок.
func TestRoutesFeedRouter(t *testing.T) {
	t.Parallel()

	cfg, err := llmconfig.Load([]byte(devConfig), nil)
	if err != nil {
		t.Fatal(err)
	}

	routes, err := cfg.Routes()
	if err != nil {
		t.Fatal(err)
	}

	router := &llm.Router{Providers: map[string]llm.Provider{"local": profiled{}}, Tasks: routes}
	needs := map[string]llm.Needs{"nameplate": {ImagesPerMessage: 1, ImagesPerRequest: 1}, "ask": {}}
	if validateErr := router.Validate(needs); validateErr != nil {
		t.Fatal(validateErr)
	}
}

type profiled struct{ llm.Provider }

func (profiled) Name() string { return "ollama" }

func (profiled) Capabilities(string) (llm.Capabilities, bool) {
	return llm.Capabilities{
		Vision: true, JSON: true, Schema: true, Strict: true, Temperature: true, Reasoning: true, MaxOutputTokens: true,
	}, true
}
