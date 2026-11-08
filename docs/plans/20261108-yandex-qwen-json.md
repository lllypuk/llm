# Яндекс: qwen3.6 в режиме json без `json_object`

## Overview

DPP и family-money переезжают на Yandex AI Studio. Единственная vision-модель Яндекса —
`qwen3.6-35b-a3b`, а с `response_format: {"type": "json_object"}` её ограниченное декодирование
выдаёт мусор вместо ответа: на 12 шильдиках DPP (замер 2026-11-08, промпт `NameplatePrompt`)
модель прочитана в 3 случаях из 12, а `ParseNameplate` не разобрал 8 ответов. Без
`response_format` получилось 11 из 12 по модели и 10 из 12 по серийнику. Примеры мусора:
`{": ": "color", "value": "red"}` и `{".{JSON_achat}Новые строки…`. С `json_schema` модель
работает (E28 DPP), с `json_object` — нет.

## Context (from discovery)

- `yandex/yandex.go`, `encode`: `ModeJSON` → `responseFormat{Type: "json_object"}` для любой модели.
- `yandex/yandex.go`, `Capabilities`: профили по имени модели без версии, `qwen3.6-35b-a3b` — `Vision`.
- `ModeJSON` в библиотеке — подсказка формы, ответ клиент не проверяет: разбор у потребителя
  защитный (README). У GigaChat ModeJSON протоколом не выражается, поэтому `JSON: false`.
- Тесты адаптера — `yandex/yandex_test.go` (`TestChatExactRequestAndResult` сверяет тело запроса).

## Решения (приняты 2026-11-08, не переигрывать)

1. `Capabilities` у `qwen3.6-35b-a3b` по-прежнему `JSON: true`. ModeJSON для этой модели уходит
   **без** `response_format`, а форму задаёт промпт потребителя. Если поставить `JSON: false`, DPP
   не сможет собрать маршрут: его фасад шлёт только `json`.
2. Исключение задаётся по имени модели без версии, как в `Capabilities`, и только для
   `qwen3.6-35b-a3b`. У `yandexgpt` и `yandexgpt-lite` `json_object` работает (проверено
   `llmcheck` 2026-11-08), и для них поведение не меняется.
3. `ModeSchema` у qwen не меняется: `json_schema` модель соблюдает.
4. Релиз — v0.7.3, исправление без изменения API. Тег ставит диспетчер.

## Development Approach

- **testing approach**: Regular. Тест на тело запроса: у `qwen3.6-35b-a3b` и у `qwen3.6-35b-a3b/latest`
  в ModeJSON нет `response_format`, а у `yandexgpt/latest` в ModeJSON он есть и равен `json_object`.
  Случай qwen в ModeSchema уже покрыт `TestChatExactRequestAndResult`.
- Задача закрывается командами `go vet ./... && go test -race ./...`,
  `go test -race -tags live ./cmd/llmcheck` и
  `go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.14.0 run --build-tags=live`
  (v2.14.0, как в CI) — всё зелёное.
- В сеть не ходить, `llmcheck` вживую не запускать.

## Implementation Steps

### Task 1: ModeJSON без json_object у qwen3.6

**Files:** `yandex/yandex.go`, `yandex/yandex_test.go`

- [x] в `encode`, ветка `llm.ModeJSON`: для модели, имя которой без версии равно `qwen3.6-35b-a3b`,
      `ResponseFormat` не ставить; для остальных моделей — как раньше
- [x] комментарий одной строкой внутри `case llm.ModeJSON`, без новых объявлений: у qwen форму
      держит промпт, потому что `json_object` даёт мусор; подробности в README
- [x] тесты из Development Approach — табличные, в стиле файла
- [x] проверки Development Approach — зелёные

### Task 2: [Final] README, CHANGELOG

**Files:** `README.md`, `CHANGELOG.md`

- [x] README, «Что делает клиент», рядом с пунктом про Ollama: у `qwen3.6-35b-a3b` ModeJSON уходит
      без `response_format`. `JSON: true` у этой модели значит «форму держит промпт», а не протокол,
      в отличие от GigaChat с `JSON: false`. Разбор у потребителя должен быть защитным.
- [x] CHANGELOG: секция `## v0.7.3` «Исправление без изменения API.» с пунктом о qwen3.6 и цифрами
      замера (3/12 → 11/12)
- [x] проверки Development Approach — зелёные

## Post-Completion

- PR на GitHub → CI зелёный → мерж → тег `v0.7.3` (диспетчер).
- После тега: DPP и family-money поднимают `github.com/lllypuk/llm` до v0.7.3.
- Живой `llmcheck` по маршруту json на qwen может дать нарушение `model`/`vision`, если ответ
  придёт в ограждении: проверка ждёт голый объект. Проверку не ослабляем, качество меряется на
  кадрах потребителя.
