package llm_test

import (
	"errors"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

// TestSpeechRequestValidate — годный запрос проходит, каждый негодный отбивается RequestError до сети.
func TestSpeechRequestValidate(t *testing.T) {
	t.Parallel()

	second := make([]byte, 32000)

	cases := []struct {
		name    string
		req     llm.SpeechRequest
		max     time.Duration
		wantErr bool
	}{
		{"годный", llm.SpeechRequest{Model: "m", SampleRate: 16000, PCM: second}, 30 * time.Second, false},
		{"ровно предел", llm.SpeechRequest{Model: "m", SampleRate: 16000, PCM: second}, time.Second, false},
		{"без предела", llm.SpeechRequest{Model: "m", SampleRate: 16000, PCM: make([]byte, 32000*120)}, 0, false},
		{"без модели", llm.SpeechRequest{SampleRate: 16000, PCM: second}, 0, true},
		{"без частоты", llm.SpeechRequest{Model: "m", PCM: second}, 0, true},
		{"отрицательная частота", llm.SpeechRequest{Model: "m", SampleRate: -1, PCM: second}, 0, true},
		{"пустая запись", llm.SpeechRequest{Model: "m", SampleRate: 16000}, 0, true},
		{"нечётная длина", llm.SpeechRequest{Model: "m", SampleRate: 16000, PCM: make([]byte, 3)}, 0, true},
		{
			"длиннее предела",
			llm.SpeechRequest{Model: "m", SampleRate: 16000, PCM: make([]byte, 32002)},
			time.Second,
			true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := tc.req.Validate(tc.max)
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("Validate() = %v, ждали nil", err)
				}

				return
			}

			var reqErr *llm.RequestError
			if !errors.As(err, &reqErr) {
				t.Fatalf("Validate() = %v, ждали *RequestError", err)
			}
		})
	}
}

// TestSpeechRequestDuration — длительность считается по сэмплам 16 бит, негодная частота даёт ноль.
func TestSpeechRequestDuration(t *testing.T) {
	t.Parallel()

	if d := (llm.SpeechRequest{SampleRate: 16000, PCM: make([]byte, 16000)}).Duration(); d != 500*time.Millisecond {
		t.Fatalf("Duration() = %v, ждали 500ms", d)
	}

	if d := (llm.SpeechRequest{PCM: make([]byte, 16000)}).Duration(); d != 0 {
		t.Fatalf("Duration() без частоты = %v, ждали 0", d)
	}
}
