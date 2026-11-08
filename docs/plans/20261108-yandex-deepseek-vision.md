# Яндекс: deepseek-v4.1-flash принимает кадры

## Overview

DPP переводит распознавание шильдиков с `qwen3.6-35b-a3b` на `deepseek-v4.1-flash` (решение владельца
2026-11-08). Замер на 52 размеченных кадрах DPP, промпт `NameplatePrompt`:

| Модель | JSON | марка | модель | серийник | медиана |
|---|---|---|---|---|---|
| qwen3.6-35b-a3b | 41/52 (11 обрывов по `length`) | 32/47 | 24/45 | 12/23 | 12 с |
| deepseek-v4.1-flash | 51/52 | 36/47 | 28/45 | 13/23 | 28 с |

`Provider.Capabilities` не знает deepseek: маршрут с кадрами отбивается классом `never`. Проба
2026-11-08: `deepseek-v4.1-flash/latest` с `response_format: json_object` отвечает корректным JSON
по схеме шильдика за 7,1 с (reasoning 312 токенов), то есть у неё `json_object` работает — в отличие
от qwen3.6.

## Context (from discovery)

- `yandex/yandex.go`, `Capabilities`: профили по имени модели без версии; кадры (`Vision`) — только у
  `qwen3.6-35b-a3b`. Комментарий над функцией это утверждает.
- `yandex/yandex.go`, `encode`: исключение «без `response_format`» задано именем qwen3.6 (v0.7.3);
  остальные модели получают `json_object`.
- `yandex/yandex_test.go`, `TestJSONModeFormat` — таблица «модель → хвост тела запроса».

## Решения (приняты 2026-11-08, не переигрывать)

1. Профиль `deepseek-v4.1-flash` (имя без версии, `deepseek-v4.1-flash/latest` — то же): общий набор
   (`JSON`, `Schema`, `Strict`, `SchemaName`, `Temperature`, `MaxOutputTokens`) плюс `Vision`.
   `Reasoning` не ставить: модель рассуждает сама (312 токенов в пробе), но профиль `Reasoning`
   у yandex означает управляемые рассуждения у открытых reasoning-моделей, а параметра для deepseek не проверяли.
2. ModeJSON у deepseek уходит с `json_object`, как у `yandexgpt*`: `encode` не меняется. `Schema: true`
   не проверялась пробой, поэтому в README и CHANGELOG не обещается.
3. Другие deepseek-модели не добавляются: замерена одна.
4. Релиз — v0.7.4, расширение профиля без изменения API. Тег ставит диспетчер.

## Development Approach

- **testing approach**: Regular. Тесты в стиле `yandex/yandex_test.go`.
- Задача закрывается командами `go vet ./... && go test -race ./...`,
  `go test -race -tags live ./cmd/llmcheck` и `go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.14.0 run --build-tags=live` (v2.14.0, как в CI) — всё зелёное.
- В сеть не ходить, `llmcheck` вживую не запускать.

## Implementation Steps

### Task 1: профиль deepseek-v4.1-flash

**Files:** `yandex/yandex.go`, `yandex/yandex_test.go`

- [x] в `Capabilities` новая ветка `case "deepseek-v4.1-flash"` с `caps.Vision = true`; док-комментарий
      над функцией поправить: кадры у qwen3.6 и deepseek
- [x] новый табличный `TestCapabilities` в `yandex/yandex_test.go` (теста профилей там нет): `qwen3.6-35b-a3b` —
      `Vision`; `deepseek-v4.1-flash` и `deepseek-v4.1-flash/latest` — `Vision`, `JSON`, `MaxOutputTokens`, без
      `Reasoning`; `yandexgpt` — без `Vision`; `gpt-oss-120b` — `Reasoning`; неизвестная (`deepseek-v9`) — `ok == false`
- [x] `TestJSONModeFormat`: строки `deepseek-v4.1-flash` и `deepseek-v4.1-flash/latest` с `json_object`; заголовок теста
      поправить под «qwen3.6 без, остальные с»
- [x] отдельный кейс рядом с `TestChatExactRequestAndResult`: запрос `deepseek-v4.1-flash/latest` с кадром и ModeJSON —
      в теле `image_url` и `response_format` `json_object`
- [x] проверки Development Approach — зелёные

### Task 2: [Final] README, CHANGELOG

**Files:** `README.md`, `CHANGELOG.md`

- [ ] README: везде, где сказано, что кадры принимает только `qwen3.6-35b-a3b`
      (`grep -n "qwen3.6\|кадр" README.md`), добавить `deepseek-v4.1-flash`; рядом с пунктом про
      `qwen3.6-35b-a3b` (строка ~54) одной фразой: у deepseek `json_object` работает, исключение — только qwen3.6.
      Таблицы или списка моделей с кадрами нет — не заводить
- [ ] `CLAUDE.md`: версию линтера в строках команд поправить с v2.12.2 на v2.14.0 (как в CI)
- [ ] CHANGELOG: секция `## v0.7.4` «Без изменения API.» — Яндекс: `deepseek-v4.1-flash` (с версией и
      без) подтверждена как vision-модель; цифры замера из Overview (JSON 41/52 → 51/52)
- [ ] проверки Development Approach — зелёные

## Post-Completion

- PR на GitHub → CI зелёный → мерж → тег `v0.7.4` (диспетчер).
- После тега: DPP поднимает `github.com/lllypuk/llm` до v0.7.4 (план `20261108-nameplate-deepseek-ocr`).
- Тариф deepseek (вход 0,3 ₽, кеш 0,075, выход 0,5 за 1K с НДС) заводит потребитель в своём `llm.json`; `pricing` библиотеки не трогаем.
