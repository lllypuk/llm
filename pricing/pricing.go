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

const tokensPerRate = 1_000_000

// Rates — цена части расхода в микроединицах валюты за миллион токенов; nil — тарифа нет.
type Rates struct {
	BillableInput *int64
	CachedInput   *int64
	Reasoning     *int64
	Output        *int64
}

// PricePlan — тариф одной ревизии, действующий с ValidFrom до ValidFrom следующей.
type PricePlan struct {
	Revision  string
	Currency  string
	ValidFrom time.Time
	Free      bool
	Rates     Rates
}

// Validate отбивает тариф, по которому оценка вышла бы неверной, а не неизвестной.
func (p PricePlan) Validate() error {
	var errs []error

	if p.Revision == "" {
		errs = append(errs, errors.New("pricing: пустая ревизия"))
	}

	if p.Currency == "" && !p.Free {
		errs = append(errs, fmt.Errorf("pricing: %s: пустая валюта", p.Revision))
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
// начавшейся не позже её старта. Разные ревизии перечисляются через запятую в порядке
// попыток, разные валюты платных ревизий не складываются — unknown; бесплатная ревизия
// валюты не навязывает.
func EstimateCall(attempts []llm.AttemptReport, plans ...PricePlan) Cost {
	var (
		total     Cost
		revisions []string
		known     int
		paid      int
		partial   bool
	)

	for _, attempt := range attempts {
		est := estimateAt(attempt, plans)
		if est.Status == StatusUnknown {
			continue
		}

		if est.Status != StatusFree {
			if paid > 0 && est.Currency != total.Currency {
				return Cost{Status: StatusUnknown}
			}

			paid++
		}

		if est.AmountMicro > math.MaxInt64-total.AmountMicro {
			return Cost{Status: StatusUnknown}
		}

		if est.Status != StatusFree || paid == 0 {
			total.Currency = est.Currency
		}

		total.AmountMicro += est.AmountMicro
		known++
		partial = partial || est.Status == StatusPartial

		if !slices.Contains(revisions, est.Revision) {
			revisions = append(revisions, est.Revision)
		}
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

// estimateAt оценивает попытку ревизией, действовавшей на её старте.
func estimateAt(attempt llm.AttemptReport, plans []PricePlan) Cost {
	plan, ok := planAt(plans, attempt.StartedAt)
	if !ok {
		return Cost{Status: StatusUnknown}
	}

	return Estimate(attempt, plan)
}

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

	return found, ok
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
	}
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
