# v0.7.0: сборка `Router` из `llmconfig` и раздел «Ломает:» в CHANGELOG

## Overview

Сборка плеча по `llmconfig.Provider.Kind` написана четыре раза: `cmd/llmcheck/main.go:295` (чат),
`cmd/llmcheck/speech.go:416` (речь), `cmd/llmcheck/ocr.go:168` (OCR) и
`../digital-property-passport/internal/llm/routes.go:451` (все шесть видов). Новый вид плеча правится
во всех, а копии уже разошлись в транспорте. Подпакет `llmbuild` собирает плечо и `*llm.Router` из
`llmconfig.Config`; `llmcheck` переходит на него в этом плане, DPP — своим.

Второе: CHANGELOG не помечал ломающих изменений — family-money на v0.2.0 при переходе упал на
`Request.Temperature` → `Request.Options.Temperature` и `Usage.InputTokens` (поле → метод), обе правки
v0.3.0. Заводится раздел `Ломает:` у версий, задним числом — у прошлых.

Не делаем: запасное плечо, обёртки наблюдателя и счётчики (остаются у потребителя), проверки задач
под потребителя (DPP требует `output.mode = json` — его правило), чтение `ca_file` в `Parse`.

## Решения (приняты 2026-10-09, не переигрывать)

1. **Сборщик в библиотеке** (codex, волна утверждена пользователем): `llmconfig` и `CLAUDE.md` прямо записывают «плечи по
   конфигу собирает потребитель, не библиотека». Принципу `docs/go-libs.md` «механизм в библиотеке,
   политика в проекте» сборщик не противоречит, если транспорт — параметр: соответствие вида
   конструктору — механизм, `http.Client`, корни и обёртки — политика.
2. **Отдельный подпакет `llmbuild`**, а не `llmconfig`: разбор конфига остаётся без адаптеров.
3. **API**:
   - `Options{PEM map[string][]byte; HTTP func(name, kind string) *http.Client}`. `PEM` — корни по имени плеча; нет
     ключа — читается `Provider.CAFile`, пустой путь — корни системы. DPP читает PEM при загрузке
     конфига и кладёт его в отпечаток настройки (`routes.go:266`), поэтому байты, а не путь. `HTTP` —
     основа клиента по имени и виду плеча: имя нужно, чтобы потребитель дал разным плечам одного
     вида разный клиент; `nil` — `&http.Client{}`.
   - `Arm(name string, p llmconfig.Provider, o Options) (Built, error)` — корни из `o.PEM[name]`, `Built{Provider; Speech;
     OCR}` — ровно одно поле не `nil`; неизвестный вид — ошибка.
   - `Router(cfg llmconfig.Config, o Options) (*llm.Router, error)` — плечи `cfg.Active()`, задачи
     `Routes()`, `SpeechRoutes()`, `OCRRoutes()`; ошибки копятся `errors.Join` с префиксом
     `providers.<имя>:` в порядке имён, как у DPP.
4. **Корни — целиком, без системных** (как в обоих копиях): плечам Сбера — `Config.CA`, остальным —
   тот же механизм: `internal/sberauth.trusting` выносится в общий `internal/` — клон транспорта основы
   от `HTTP`, меняются только `RootCAs` и `InsecureSkipVerify=false`, прочий `TLSClientConfig` основы не
   затирается. Транспорт основы не `*http.Transport` при заданных корнях — ошибка, а не молчаливая подмена.
5. **`Ломает:`** — подраздел версии со списком «было → стало» по символам; нет ломающих — подраздела нет.
   Правило — строкой в `CLAUDE.md` (раздел «Процесс»).
6. **Отменяется правило «плечи по конфигу собирает потребитель, не библиотека»** (`CLAUDE.md:37`,
   док `llmconfig` `config.go:3`, `README.md:60`). Новый текст: механизм сборки — в библиотеке
   (`llmbuild`), транспорт и политика — у потребителя.

## Context (from discovery)

- `llmconfig/config.go:21-28` — виды; `:397-405` — проверка полей по виду; `Active()`, `Routes()`,
  `SpeechRoutes()`, `OCRRoutes()`.
- `cmd/llmcheck`: `build` (`main.go:286`, ещё отдаёт `oauthTarget` GigaChat — он собирается из
  конфига без адаптера и остаётся в `llmcheck`), `loadCA`, `trusting`; `buildSpeech`, `buildOCR`;
  обёртки `countedSpeech`, `countedOCR` надеваются после сборки.
- DPP `internal/llm/routes.go:434-527` — `buildProvider` и `transport`: Сберу `&http.Client{}` +
  `CA`, остальным клон транспорта с корнями.
- Образец среза с README и CHANGELOG — `docs/plans/completed/20260925-ocr-route.md`.

## Development Approach

- **testing approach**: TDD.
- **CRITICAL: каждая задача кончается тестами, они зелёные до следующей.**
- Перед каждым коммитом: `go vet ./... && go test -race ./...` и
  `golangci-lint run --build-tags=live` (v2.12.2; локально —
  `go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.12.2 run --build-tags=live`).
- Сеть и ключи не нужны: `go run -tags live ./cmd/llmcheck` не запускать.
- Коммиты на русском: `feat:`, `refactor:`, `docs:`; комментарии — по `~/.claude/CLAUDE.md`.

## Testing Strategy

- `llmbuild` — юнит-тесты без сети: вид → тип плеча (`Name()`), корни (самоподписанный PEM попадает в
  `RootCAs` клиента; системных нет), ошибки конфигурации.
- `llmcheck` — существующие `build_test.go`, `speech_test.go`, `ocr_test.go`, `TestLiveBuild`
  зелёные без правки ожиданий.

## Progress Tracking

`[x]` — сделано, ➕ — найдено по ходу, ⚠️ — блокер.

## Implementation Steps

### Task 1: `llmbuild.Arm`

**Files:** `llmbuild/llmbuild.go`, `llmbuild/llmbuild_test.go`

- [x] `Options`, `Built`, `Arm` по Решениям 3–4; корни из `o.PEM[name]` или `CAFile`; `trusting` — в общий `internal/`
- [x] тесты: шесть видов дают своё поле `Built`; неизвестный вид; битый PEM и пустой файл корней;
  с корнями у не-Сбера клиент несёт `RootCAs` из PEM, прочие TLS-настройки основы сохраняются; без корней — основа `HTTP` как есть
- [x] `go test -race ./llmbuild/` — зелёный

### Task 2: `llmbuild.Router`

**Files:** `llmbuild/router.go`, `llmbuild/router_test.go`

- [x] `Router` по Решению 3; `Validate(nil)` собранного роутера проходит
- [x] тесты: конфиг с чатом, речью и OCR — плечи в своих картах, задачи из `Routes*`; неактивное плечо
  не собирается (его секреты не нужны); две ошибки плеч — обе в ответе, по порядку имён
- [x] `go test -race ./...` — зелёный

### Task 3: `cmd/llmcheck` на `llmbuild`

**Files:** `cmd/llmcheck/main.go`, `cmd/llmcheck/speech.go`, `cmd/llmcheck/ocr.go`

- [x] `build`, `buildSpeech`, `buildOCR` заменить вызовом `llmbuild.Arm` с проверкой нужного поля
  `Built` (прежние тексты «вид плеча %q не распознаёт речь/текст» сохранить); `oauthTarget` собирать из `llmconfig.Provider`
  с тем же клиентом на корнях `CAFile`, что сейчас (`main.go:310`): без них отдельная проверка OAuth
  упрётся в сертификат Сбера — `loadCA` и `trusting` остаются для неё
- [x] `go test -race -tags live ./cmd/llmcheck` и `go test -race ./...` — зелёные

### Task 4: раздел «Ломает:» задним числом

**Files:** `CHANGELOG.md`

- [x] сравнить экспортируемый API соседних тегов (`git worktree add` на тег, `go doc -all` по
  пакетам, diff): v0.1.0→v0.2.0 … v0.6.0→v0.6.1
- [x] у затронутых версий — подраздел `Ломает:`; в v0.3.0 обязательно `Request.Temperature` →
  `Request.Options.Temperature` и `Usage.InputTokens` поле → метод (части `Usage` непересекающиеся);
  в v0.4.0 — обязательный `Provider.AttemptOverhead` для своих плеч
- [x] `go test -race ./...` — зелёный (CHANGELOG тестами не читается — проверка, что ничего не задето)

### Task 5: Verify acceptance criteria

- [x] `switch` по `Kind` вне `llmconfig` и `llmbuild` не осталось (`grep -rn 'switch p.Kind' --include='*.go' . | grep -v '^./llm\(config\|build\)/'` — вывод пустой)
- [x] `go vet ./... && go test -race ./...`, `go test -race -tags live ./cmd/llmcheck`
- [x] `golangci-lint run --build-tags=live` — 0 замечаний; зависимостей вне stdlib нет

### Task 6: [Final] Документы v0.7.0

**Files:** `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `llmconfig/config.go`

- [ ] `CHANGELOG.md`: `## v0.7.0` — `llmbuild`, `llmcheck` на нём; `Ломает:` нет
- [ ] `README.md`: короткий раздел о `llmbuild` рядом с `llmconfig` — пример `Router(cfg, Options{})`;
  `Router` принимает развёрнутый конфиг (`Load` или `Expand`); транспорт и обёртки остаются у потребителя
- [ ] `CLAUDE.md`: строку «Плечи по конфигу собирает потребитель…» заменить текстом Решения 6; в «Процесс» — правило `Ломает:` одной
  строкой; в перечень «Новое плечо затрагивает» добавить `llmbuild`
- [ ] док пакета `llmconfig` и `README.md:60`: правило «собирает потребитель» → текст Решения 6
- [ ] последний коммит — `docs: v0.7.0 — CHANGELOG, README и раздел «Ломает:»`

## Post-Completion

- Аннотированный тег `v0.7.0: сборка Router из конфига` и пуш — диспетчер после мержа.
- DPP: план перехода на v0.7.0 — `buildProvider` и `transport` (`internal/llm/routes.go:432-527`)
  заменяются `llmbuild.Router` с `Options{PEM: s.CA}`; проверки DPP (`output.mode`, `onlyTask`,
  тарифы) остаются; бюджет `commentBudget` снижается тем же коммитом.
- family-money: переход с v0.2.0 — своим планом по разделам `Ломает:` v0.3.0–v0.7.0.
