# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Библиотека `github.com/lllypuk/llm` — вызов модели, распознавание речи и текста одним контрактом поверх
нескольких поставщиков. Поведение клиента (классы отказа, срок на попытку, расход, маршруты) описано в
`README.md` — он и есть спецификация; правка поведения правит и его, и `CHANGELOG.md`.

## Команды

```sh
go vet ./...
go test -race ./...                          # включает TestLiveBuild: вложенный go test -tags live без сети
go test -race -tags live ./cmd/llmcheck      # тесты llmcheck напрямую
go test -race -run TestName ./gigachat       # один тест
golangci-lint run --build-tags=live          # v2.12.2, как в CI; локально может не стоять:
go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.12.2 run --build-tags=live
```

Без `--build-tags=live` линтер не видит `cmd/llmcheck` — код команды под тегом `live`.
`go run -tags live ./cmd/llmcheck` ходит в сеть на деньги аккаунта; запускать только по просьбе.

## Устройство

- Корень — три независимых контракта плеча: `Provider` (чат, `llm.go`), `Transcriber` (`speech.go`),
  `Recognizer` (`ocr.go`), у каждого свой маршрут (`router.go`, `speech_route.go`, `ocr_route.go`).
  Пауза, срок и классы отказа общие — `client.go`, `errors.go`; цикл попыток у каждого контракта свой
  намеренно. Отчёты и наблюдатель — `observe.go`. `Router` общий: контракт добавляет в него пару карт
  (плечи и задачи), проверку в `Validate` и `Resolve*`; чатовые `Provider` и `Resolve` не меняются.
  OCR сделан по образцу речи, следующий контракт — по тому же.
- Подпакеты-поставщики (`ollama`, `gigachat`, `yandex`, `salutespeech`) знают только протокол и
  переводят отказ в типы корня (`*StatusError`, `*ResponseError`, `*RequestError`, `*WarnedError`) —
  от типа зависят класс повтора и попадание в расход.
- `internal/httpjson` — общий HTTP+JSON с ограниченным телом; wire-структуры остаются в адаптерах.
  `internal/sberauth` — OAuth Сбера, общий у GigaChat и SaluteSpeech.
- `llmconfig` — строгий разбор файла маршрутов; вид плеча (`Kind*`) плюс проверки его раздела.
  Плечи по конфигу собирает потребитель (и `cmd/llmcheck`), не библиотека.
- `pricing` — оценка стоимости по отчётам попыток; неизвестное — `unknown`, никогда не ноль.
- Зависимостей вне stdlib нет — новая требует обоснования.

Новое плечо или контракт затрагивает сразу: адаптер, вид в `llmconfig`, тариф в `pricing`, подкоманду
или проверку в `cmd/llmcheck`, README и CHANGELOG. Образец такого среза — план
`docs/plans/completed/20260925-ocr-route.md`.

## Процесс

- Планы — `docs/plans/`, выполненные уезжают в `completed/`. Релиз — секция версии в `CHANGELOG.md`;
  тег ставит диспетчер после мержа, не воркер.
- Линтер — общий конфиг библиотек `lllypuk` без проектных исключений: правится код, а не
  `.golangci.yml`. В нём запрещены `log` вне `main.go` и `math/rand` вне тестов.
