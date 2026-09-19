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

// Rates — цена части расхода в микроединицах валюты за миллион токенов, у записи — за секунду; nil — тарифа нет.
type Rates struct {
	BillableInput *int64
	CachedInput   *int64
	Reasoning     *int64
	Output        *int64
	AudioSecond   *int64
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
	parts := []struct {
		tokens int
		rate   *int64
	}{
		{u.BillableInput, plan.Rates.BillableInput},
		{u.CachedInput, plan.Rates.CachedInput},
		{u.Reasoning, plan.Rates.Reasoning},
		{u.Output, plan.Rates.Output},
	}

	var (
		sum      int64
		counted  bool
		complete = u.Known
	)

	for _, part := range parts {
		switch {
		case part.tokens < 0:
			return unknown
		case part.tokens == 0:
			continue
		case part.rate == nil:
			complete = false

			continue
		}

		cost, ok := mul(int64(part.tokens), *part.rate)
		if !ok || cost > math.MaxInt64-sum {
			return unknown
		}

		sum += cost
		counted = true
	}

	switch {
	case attempt.AudioMillis < 0:
		return unknown
	case attempt.AudioMillis == 0:
	case plan.Rates.AudioSecond == nil:
		complete = false
	default:
		cost, ok := audioCost(attempt.AudioMillis, plan.audioStep(), *plan.Rates.AudioSecond)
		if !ok || cost > math.MaxInt64-sum {
			return unknown
		}

		sum += cost
		counted = true
	}

	if !counted && !complete {
		return unknown
	}

	priced.AmountMicro = ceilDiv(sum, tokensPerRate)
	priced.Status = StatusEstimated

	if !complete {
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
