package pricing_test

import (
	"math"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/pricing"
)

func month(m time.Month) time.Time {
	return time.Date(2026, m, 1, 0, 0, 0, 0, time.UTC)
}

func attempt(at time.Time, u llm.Usage) llm.AttemptReport {
	return llm.AttemptReport{StartedAt: at, Usage: u}
}

func rub(revision string, from time.Time, rates pricing.Rates) pricing.PricePlan {
	return pricing.PricePlan{Revision: revision, Currency: "RUB", ValidFrom: from, Rates: rates}
}

func TestEstimateGigaChatCachePricedApart(t *testing.T) {
	plan := rub("giga-2026-09", month(time.September), pricing.Rates{
		BillableInput: new(int64(1_500_000_000)),
		CachedInput:   new(int64(150_000_000)),
		Output:        new(int64(1_500_000_000)),
	})
	u := llm.Usage{BillableInput: 1000, CachedInput: 3000, Output: 200, Known: true}

	got := pricing.Estimate(attempt(month(time.September).Add(time.Hour), u), plan)

	want := pricing.Cost{AmountMicro: 2_250_000, Currency: "RUB", Revision: "giga-2026-09", Status: "estimated"}
	if got != want {
		t.Fatalf("Estimate = %+v, want %+v", got, want)
	}
}

func TestEstimateYandexReasoningAtItsRate(t *testing.T) {
	rates := pricing.Rates{
		BillableInput: new(int64(400_000_000)),
		Output:        new(int64(400_000_000)),
	}
	u := llm.Usage{BillableInput: 500, Reasoning: 1000, Output: 250, Known: true}

	got := pricing.Estimate(attempt(month(time.September), u), rub("ya", month(time.September), rates))
	if got.Status != pricing.StatusPartial || got.AmountMicro != 300_000 {
		t.Fatalf("без тарифа рассуждений = %+v, want partial 300000 — нижняя граница", got)
	}

	rates.Reasoning = new(int64(400_000_000))

	got = pricing.Estimate(attempt(month(time.September), u), rub("ya", month(time.September), rates))
	if got.Status != pricing.StatusEstimated || got.AmountMicro != 700_000 {
		t.Fatalf("с тарифом рассуждений = %+v, want estimated 700000", got)
	}
}

func TestEstimateRoundsUp(t *testing.T) {
	plan := rub("r", month(time.September), pricing.Rates{Output: new(int64(1))})

	got := pricing.Estimate(attempt(month(time.September), llm.Usage{Output: 1, Known: true}), plan)
	if got.AmountMicro != 1 {
		t.Fatalf("AmountMicro = %d, want 1: доля микроединицы округляется вверх", got.AmountMicro)
	}
}

func TestEstimateUnknown(t *testing.T) {
	rates := pricing.Rates{BillableInput: new(int64(1)), Output: new(int64(1))}
	known := llm.Usage{BillableInput: 10, Output: 10, Known: true}

	cases := []struct {
		name    string
		attempt llm.AttemptReport
		plan    pricing.PricePlan
	}{
		{"расход не сообщён", attempt(month(time.September), llm.Usage{}), rub("r", month(time.September), rates)},
		{
			"ни одного тарифа на ненулевые части",
			attempt(month(time.September), known),
			rub("r", month(time.September), pricing.Rates{}),
		},
		{
			"попытка раньше тарифа",
			attempt(month(time.September).Add(-time.Second), known),
			rub("r", month(time.September), rates),
		},
		{"тариф без валюты", attempt(month(time.September), known), pricing.PricePlan{Revision: "r", Rates: rates}},
		{"тариф без ревизии", attempt(month(time.September), known), rub("", month(time.September), rates)},
		{
			"отрицательная цена",
			attempt(month(time.September), known),
			rub("r", month(time.September), pricing.Rates{BillableInput: new(int64(-1)), Output: new(int64(1))}),
		},
		{
			"отрицательный счётчик",
			attempt(month(time.September), llm.Usage{Output: -1, Known: true}),
			rub("r", month(time.September), rates),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pricing.Estimate(tc.attempt, tc.plan); got != (pricing.Cost{Status: "unknown"}) {
				t.Fatalf("Estimate = %+v, want unknown без суммы", got)
			}
		})
	}
}

func TestEstimatePartialWhenUsageIncomplete(t *testing.T) {
	plan := rub("r", month(time.September), pricing.Rates{BillableInput: new(int64(1_000_000))})

	got := pricing.Estimate(attempt(month(time.September), llm.Usage{BillableInput: 7}), plan)
	if got.Status != pricing.StatusPartial || got.AmountMicro != 7 {
		t.Fatalf("Estimate = %+v, want partial 7", got)
	}
}

func TestEstimateFree(t *testing.T) {
	plan := pricing.PricePlan{Revision: "ollama", Free: true}

	got := pricing.Estimate(attempt(month(time.September), llm.Usage{}), plan)
	if got != (pricing.Cost{Revision: "ollama", Status: "free"}) {
		t.Fatalf("Estimate = %+v, want free: бесплатному неизвестный расход не мешает", got)
	}

	plan.Rates.Output = new(int64(1))
	if plan.Validate() == nil {
		t.Fatal("Validate: бесплатный тариф с ценой принят")
	}
}

func TestEstimateOverflowIsUnknown(t *testing.T) {
	cases := []struct {
		name  string
		rates pricing.Rates
		usage llm.Usage
	}{
		{"произведение", pricing.Rates{Output: new(int64(math.MaxInt64 / 2))}, llm.Usage{Output: 3, Known: true}},
		{
			"сумма частей",
			pricing.Rates{BillableInput: new(int64(math.MaxInt64 / 2)), Output: new(int64(math.MaxInt64 / 2))},
			llm.Usage{BillableInput: 2, Output: 1, Known: true},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := pricing.Estimate(attempt(month(time.September), tc.usage), rub("r", month(time.September), tc.rates))
			if got.Status != pricing.StatusUnknown {
				t.Fatalf("Estimate = %+v, want unknown", got)
			}
		})
	}
}

func TestEstimateCallRevisionChangesBetweenAttempts(t *testing.T) {
	old := rub("2026-09", month(time.September), pricing.Rates{Output: new(int64(1_000_000))})
	next := rub("2026-10", month(time.October), pricing.Rates{Output: new(int64(2_000_000))})
	u := llm.Usage{Output: 100, Known: true}

	got := pricing.EstimateCall([]llm.AttemptReport{
		attempt(month(time.October).Add(-time.Second), u),
		attempt(month(time.October).Add(time.Second), u),
	}, next, old)

	want := pricing.Cost{AmountMicro: 300, Currency: "RUB", Revision: "2026-09,2026-10", Status: "estimated"}
	if got != want {
		t.Fatalf("EstimateCall = %+v, want %+v", got, want)
	}
}

func TestEstimateCallStatus(t *testing.T) {
	priced := rub("r", month(time.September), pricing.Rates{Output: new(int64(1_000_000))})
	free := pricing.PricePlan{Revision: "f", Free: true}
	known := attempt(month(time.September), llm.Usage{Output: 5, Known: true})
	silent := attempt(month(time.September), llm.Usage{})

	cases := []struct {
		name     string
		attempts []llm.AttemptReport
		plans    []pricing.PricePlan
		want     pricing.Cost
	}{
		{"без попыток", nil, []pricing.PricePlan{priced}, pricing.Cost{Status: "unknown"}},
		{"без тарифа", []llm.AttemptReport{known}, nil, pricing.Cost{Status: "unknown"}},
		{
			"сумма известных при неизвестной попытке",
			[]llm.AttemptReport{silent, known},
			[]pricing.PricePlan{priced},
			pricing.Cost{AmountMicro: 5, Currency: "RUB", Revision: "r", Status: "partial"},
		},
		{
			"бесплатная и платная ревизии",
			[]llm.AttemptReport{known, attempt(month(time.October), llm.Usage{Output: 5, Known: true})},
			[]pricing.PricePlan{free, rub("r2", month(time.October), pricing.Rates{Output: new(int64(1_000_000))})},
			pricing.Cost{AmountMicro: 5, Currency: "RUB", Revision: "f,r2", Status: "estimated"},
		},
		{
			"все бесплатные",
			[]llm.AttemptReport{silent, known},
			[]pricing.PricePlan{free},
			pricing.Cost{Revision: "f", Status: "free"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pricing.EstimateCall(tc.attempts, tc.plans...); got != tc.want {
				t.Fatalf("EstimateCall = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestEstimateCallCurrencyMismatchIsUnknown(t *testing.T) {
	rates := pricing.Rates{Output: new(int64(1_000_000))}
	usd := pricing.PricePlan{Revision: "usd", Currency: "USD", ValidFrom: month(time.October), Rates: rates}
	u := llm.Usage{Output: 1, Known: true}

	got := pricing.EstimateCall(
		[]llm.AttemptReport{attempt(month(time.September), u), attempt(month(time.October), u)},
		rub("rub", month(time.September), rates), usd,
	)
	if got != (pricing.Cost{Status: "unknown"}) {
		t.Fatalf("EstimateCall = %+v, want unknown: валюты не складываются", got)
	}
}
