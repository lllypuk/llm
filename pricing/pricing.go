// Package pricing оценивает стоимость вызова по расходу попыток и тарифу потребителя.
// Оценка — не счёт поставщика: тарифы, валюту и их ревизии держит потребитель.
package pricing

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/lllypuk/llm"
)

// Status — насколько оценке можно верить.
type Status string

// Состояния оценки.
const (
	StatusEstimated Status = "estimated"
	StatusPartial   Status = "partial"
	StatusUnknown   Status = "unknown"
	StatusFree      Status = "free"
)

const (
	tokensPerRate  = 1_000_000
	millisPerRate  = 1_000
	defaultAudioMs = 1_000
)

// Rates — цена части расхода в микроединицах валюты за миллион токенов, у записи — за секунду,
// у OCR — за страницу; nil — тарифа нет.
type Rates struct {
	BillableInput *int64
	CachedInput   *int64
	Reasoning     *int64
	Output        *int64
	AudioSecond   *int64
	Page          *int64
}

// PricePlan — тариф одной ревизии, действующий с ValidFrom до ValidFrom следующей.
// AudioStep — шаг, до которого запись округляется вверх; ноль — секунда.
type PricePlan struct {
	Revision  string
	Currency  string
	ValidFrom time.Time
	Free      bool
	Rates     Rates
	AudioStep time.Duration
}

// Validate отбивает тариф, по которому оценка вышла бы неверной, а не неизвестной.
func (p PricePlan) Validate() error {
	var errs []error

	switch {
	case p.Revision == "":
		errs = append(errs, errors.New("pricing: пустая ревизия"))
	case strings.Contains(p.Revision, ","):
		errs = append(errs, fmt.Errorf("pricing: %s: запятая в ревизии — разделитель списка ревизий", p.Revision))
	}

	if p.Currency == "" && !p.Free {
		errs = append(errs, fmt.Errorf("pricing: %s: пустая валюта", p.Revision))
	}

	if p.AudioStep < 0 || p.AudioStep%time.Millisecond != 0 {
		errs = append(errs, fmt.Errorf("pricing: %s: шаг записи не кратен миллисекунде", p.Revision))
	}

	for _, r := range p.Rates.named() {
		switch {
		case r.rate == nil:
		case p.Free:
			errs = append(errs, fmt.Errorf("pricing: %s: бесплатный тариф с ценой %s", p.Revision, r.name))
		case *r.rate < 0:
			errs = append(errs, fmt.Errorf("pricing: %s: отрицательная цена %s", p.Revision, r.name))
		}
	}

	return errors.Join(errs...)
}

// Cost — оценка стоимости попытки или вызова. AmountMicro при partial — нижняя граница.
type Cost struct {
	AmountMicro int64
	Currency    string
	Revision    string
	Status      Status
}

// Estimate оценивает попытку тарифом plan. Попытка раньше ValidFrom, негодный тариф
// и переполнение дают unknown, а не ноль.
func Estimate(attempt llm.AttemptReport, plan PricePlan) Cost {
	unknown := Cost{Status: StatusUnknown}

	if plan.Validate() != nil || attempt.StartedAt.Before(plan.ValidFrom) {
		return unknown
	}

	priced := Cost{Currency: plan.Currency, Revision: plan.Revision}

	if plan.Free {
		priced.Status = StatusFree

		return priced
	}

	u := attempt.Usage
	t := tally{complete: u.Known}
	audio := func(millis, rate int64) (int64, bool) { return audioCost(millis, plan.audioStep(), rate) }

	ok := t.add(int64(u.BillableInput), plan.Rates.BillableInput, tokenCost) &&
		t.add(int64(u.CachedInput), plan.Rates.CachedInput, tokenCost) &&
		t.add(int64(u.Reasoning), plan.Rates.Reasoning, tokenCost) &&
		t.add(int64(u.Output), plan.Rates.Output, tokenCost) &&
		t.add(attempt.AudioMillis, plan.Rates.AudioSecond, audio) &&
		t.add(int64(attempt.Pages), plan.Rates.Page, pageCost)
	if !ok {
		return unknown
	}

	if !t.counted && !t.complete {
		return unknown
	}

	priced.AmountMicro = ceilDiv(t.sum, tokensPerRate)
	priced.Status = StatusEstimated

	if !t.complete {
		priced.Status = StatusPartial
	}

	return priced
}

// EstimateCall складывает оценки попыток; попытке достаётся тариф последней ревизии,
// начавшейся не позже её старта; негодная ревизия делает попытку неизвестной. Разные ревизии перечисляются через запятую в порядке попыток.
// Разные валюты платных ревизий — unknown, даже у попытки с неизвестным расходом; бесплатная
// валюты не навязывает. Ожидает тарифы одного маршрута с различными ValidFrom.
func EstimateCall(attempts []llm.AttemptReport, plans ...PricePlan) Cost {
	var (
		total     Cost
		revisions []string
		known     int
		paid      int
		partial   bool
	)

	for _, attempt := range attempts {
		plan, ok := planAt(plans, attempt.StartedAt)
		if !ok {
			continue
		}

		if !mergeCurrency(&total, &paid, plan) {
			return Cost{Status: StatusUnknown}
		}

		if !slices.Contains(revisions, plan.Revision) {
			revisions = append(revisions, plan.Revision)
		}

		est := Estimate(attempt, plan)
		if est.Status == StatusUnknown {
			continue
		}

		if est.AmountMicro > math.MaxInt64-total.AmountMicro {
			return Cost{Status: StatusUnknown}
		}

		total.AmountMicro += est.AmountMicro
		known++
		partial = partial || est.Status == StatusPartial
	}

	total.Revision = strings.Join(revisions, ",")

	switch {
	case known == 0:
		return Cost{Status: StatusUnknown}
	case known < len(attempts) || partial:
		total.Status = StatusPartial
	case paid == 0:
		total.Status = StatusFree
	default:
		total.Status = StatusEstimated
	}

	return total
}

// mergeCurrency переносит валюту ревизии в сумму; ложь — платные ревизии расходятся валютой.
// Бесплатная ревизия валюту платной не перетирает.
func mergeCurrency(total *Cost, paid *int, plan PricePlan) bool {
	if plan.Free {
		if *paid == 0 {
			total.Currency = plan.Currency
		}

		return true
	}

	if *paid > 0 && plan.Currency != total.Currency {
		return false
	}

	*paid++
	total.Currency = plan.Currency

	return true
}

// planAt — последняя ревизия, начавшаяся не позже at; негодная не выбирается, и старшая за неё не отвечает.
func planAt(plans []PricePlan, at time.Time) (PricePlan, bool) {
	var (
		found PricePlan
		ok    bool
	)

	for _, p := range plans {
		if !at.Before(p.ValidFrom) && (!ok || p.ValidFrom.After(found.ValidFrom)) {
			found, ok = p, true
		}
	}

	return found, ok && found.Validate() == nil
}

type namedRate struct {
	name string
	rate *int64
}

func (r Rates) named() []namedRate {
	return []namedRate{
		{"billable_input", r.BillableInput},
		{"cached_input", r.CachedInput},
		{"reasoning", r.Reasoning},
		{"output", r.Output},
		{"audio_second", r.AudioSecond},
		{"page", r.Page},
	}
}

func (p PricePlan) audioStep() int64 {
	if p.AudioStep == 0 {
		return defaultAudioMs
	}

	return p.AudioStep.Milliseconds()
}

// audioCost — запись, округлённая вверх до шага, в масштабе суммы токенов: миллионных долях микроединицы.
func audioCost(millis, step, rate int64) (int64, bool) {
	billed, ok := mul(ceilDiv(millis, step), step)
	if !ok {
		return 0, false
	}

	perMilli, ok := mul(billed, rate)
	if !ok {
		return 0, false
	}

	return mul(perMilli, tokensPerRate/millisPerRate)
}

// tally — сумма частей расхода в миллионных долях микроединицы.
type tally struct {
	sum      int64
	counted  bool
	complete bool
}

// add прибавляет часть расхода; ложь — отрицательный расход или переполнение, оценка неизвестна.
func (t *tally) add(units int64, rate *int64, cost func(units, rate int64) (int64, bool)) bool {
	switch {
	case units < 0:
		return false
	case units == 0:
		return true
	case rate == nil:
		t.complete = false

		return true
	}

	c, ok := cost(units, *rate)
	if !ok || c > math.MaxInt64-t.sum {
		return false
	}

	t.sum += c
	t.counted = true

	return true
}

func tokenCost(tokens, rate int64) (int64, bool) {
	return mul(tokens, rate)
}

// pageCost — страницы в масштабе суммы токенов, как [audioCost].
func pageCost(pages, rate int64) (int64, bool) {
	perPage, ok := mul(pages, rate)
	if !ok {
		return 0, false
	}

	return mul(perPage, tokensPerRate)
}

func mul(a, b int64) (int64, bool) {
	if a != 0 && b > math.MaxInt64/a {
		return 0, false
	}

	return a * b, true
}

// ceilDiv округляет вверх: оценка расхода не должна обещать меньше потраченного.
func ceilDiv(a, b int64) int64 {
	q := a / b
	if a%b != 0 {
		q++
	}

	return q
}
