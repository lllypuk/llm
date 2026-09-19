package yandex_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/yandex"
)

// sttServer — `/speech/v1/stt:recognize` с одним ответом; запоминает запросы.
type sttServer struct {
	status int
	body   string
	header http.Header

	mu   sync.Mutex
	reqs []*http.Request
	pcm  [][]byte
}

func (s *sttServer) start(t *testing.T) string {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.reqs = append(s.reqs, r)
		s.pcm = append(s.pcm, raw)
		s.mu.Unlock()

		for k, v := range s.header {
			w.Header()[k] = v
		}

		w.Header().Set("X-Request-Id", "stt-1")
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.body)
	}))
	t.Cleanup(srv.Close)

	return srv.URL + "/speech/v1/"
}

func (s *sttServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return len(s.reqs)
}

func speech(t *testing.T, endpoint string, creds yandex.CredentialSource) *yandex.Speech {
	t.Helper()

	p, err := yandex.NewSpeech(yandex.SpeechConfig{Endpoint: endpoint, Folder: "b1g", Credentials: creds})
	if err != nil {
		t.Fatal(err)
	}

	return p
}

// pcm — секунды тишины LPCM 16 кГц.
func pcm(seconds float64) []byte {
	return make([]byte, int(seconds*16000)*2)
}

func sttRequest(audio []byte) llm.SpeechRequest {
	return llm.SpeechRequest{Model: "general", Language: "ru-RU", SampleRate: 16000, PCM: audio}
}

// TestSpeechExactRequestAndResult — адрес, query, подпись и тело байт в байт; ответ в Transcript.
func TestSpeechExactRequestAndResult(t *testing.T) {
	t.Parallel()

	s := &sttServer{status: http.StatusOK, body: `{"result":" поменял фильтр "}`}
	p := speech(t, s.start(t), yandex.APIKey("k"))

	audio := pcm(1.5)
	audio[0], audio[len(audio)-1] = 7, 9

	got, err := p.Transcribe(context.Background(), sttRequest(audio))
	if err != nil {
		t.Fatal(err)
	}

	want := llm.Transcript{Text: "поменял фильтр", Model: "general", RequestID: "stt-1", AudioMillis: 1500}
	if got != want {
		t.Errorf("результат %+v, ждали %+v", got, want)
	}

	r := s.reqs[0]
	if r.Method != http.MethodPost || r.URL.Path != "/speech/v1/stt:recognize" {
		t.Errorf("запрос %s %s", r.Method, r.URL.Path)
	}

	wantQuery := "folderId=b1g&format=lpcm&lang=ru-RU&sampleRateHertz=16000&topic=general"
	if r.URL.RawQuery != wantQuery {
		t.Errorf("query %q, ждали %q", r.URL.RawQuery, wantQuery)
	}

	if a := r.Header.Get("Authorization"); a != "Api-Key k" {
		t.Errorf("подпись %q", a)
	}

	if ct := r.Header.Get("Content-Type"); ct != "application/octet-stream" {
		t.Errorf("Content-Type %q", ct)
	}

	if !bytes.Equal(s.pcm[0], audio) {
		t.Error("тело не совпало с PCM")
	}
}

// TestSpeechEmptyResultIsSuccess — речи не расслышали: успех с пустым текстом, а не отказ.
func TestSpeechEmptyResultIsSuccess(t *testing.T) {
	t.Parallel()

	for name, body := range map[string]string{"пустой": `{"result":""}`, "без поля": `{}`} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &sttServer{status: http.StatusOK, body: body}

			got, err := speech(t, s.start(t), yandex.APIKey("k")).Transcribe(context.Background(), sttRequest(pcm(1)))
			if err != nil || got.Text != "" || got.AudioMillis != 1000 {
				t.Fatalf("результат %+v, %v", got, err)
			}
		})
	}
}

// TestSpeechStatus — не-2xx становится StatusError с текстом конверта, Retry-After и идентификатором.
func TestSpeechStatus(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		status int
		body   string
		header http.Header
		want   llm.StatusError
	}{
		"401": {
			http.StatusUnauthorized, `{"error_code":"UNAUTHORIZED","error_message":"bad key"}`, nil,
			llm.StatusError{Status: http.StatusUnauthorized, Message: "bad key", RequestID: "stt-1"},
		},
		"429": {
			http.StatusTooManyRequests, `{"error_code":"TOO_MANY_REQUESTS"}`, http.Header{"Retry-After": {"3"}},
			llm.StatusError{
				Status: http.StatusTooManyRequests, Message: "TOO_MANY_REQUESTS",
				RetryAfter: 3 * time.Second, RequestID: "stt-1",
			},
		},
		"503": {
			http.StatusServiceUnavailable, "upstream down", nil,
			llm.StatusError{Status: http.StatusServiceUnavailable, Message: "upstream down", RequestID: "stt-1"},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &sttServer{status: tc.status, body: tc.body, header: tc.header}

			_, err := speech(t, s.start(t), yandex.APIKey("k")).Transcribe(context.Background(), sttRequest(pcm(1)))

			var st *llm.StatusError
			if !errors.As(err, &st) || *st != tc.want {
				t.Fatalf("отказ %v (%+v), ждали %+v", err, st, tc.want)
			}
		})
	}
}

// TestSpeechCancelled — отмена контекста обрывает висящий запрос и остаётся отменой.
func TestSpeechCancelled(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := speech(t, srv.URL, yandex.APIKey("k")).Transcribe(ctx, sttRequest(pcm(1)))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("отказ %v", err)
	}
}

// TestSpeechRejectedBeforeNetwork — длинная запись и негодный запрос не доходят до сервера, как и отказ IAM.
func TestSpeechRejectedBeforeNetwork(t *testing.T) {
	t.Parallel()

	boom := errors.New("метаданные недоступны")

	cases := map[string]struct {
		creds yandex.CredentialSource
		req   llm.SpeechRequest
	}{
		"длиннее 30 с":  {yandex.APIKey("k"), sttRequest(pcm(30.5))},
		"без модели":    {yandex.APIKey("k"), llm.SpeechRequest{SampleRate: 16000, PCM: pcm(1)}},
		"пустая запись": {yandex.APIKey("k"), sttRequest(nil)},
		"отказ IAM": {
			yandex.IAMToken(func(context.Context) (string, error) { return "", boom }),
			sttRequest(pcm(1)),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &sttServer{status: http.StatusOK, body: `{"result":"x"}`}

			_, err := speech(t, s.start(t), tc.creds).Transcribe(context.Background(), tc.req)

			var reqErr *llm.RequestError

			var phase *llm.PhaseError
			if !errors.As(err, &reqErr) && (!errors.As(err, &phase) || phase.Phase != llm.PhaseAuth) {
				t.Errorf("отказ %v", err)
			}

			if s.count() != 0 {
				t.Error("запрос ушёл в сеть")
			}
		})
	}
}

// TestSpeechCapabilities — предел у известных моделей, неизвестная не подтверждается.
func TestSpeechCapabilities(t *testing.T) {
	t.Parallel()

	p := speech(t, "http://x", yandex.APIKey("k"))

	if caps, ok := p.SpeechCapabilities("general:rc"); !ok || caps.MaxAudio != 30*time.Second {
		t.Errorf("general:rc %+v %v", caps, ok)
	}

	if _, ok := p.SpeechCapabilities("yandexgpt"); ok {
		t.Error("чатовая модель подтверждена")
	}

	if p.Name() != yandex.SpeechName {
		t.Error(p.Name())
	}
}

// TestNewSpeechRejectsIncompleteConfig — без адреса, каталога и подписи плечо не собирается.
func TestNewSpeechRejectsIncompleteConfig(t *testing.T) {
	t.Parallel()

	full := yandex.SpeechConfig{Endpoint: "http://x", Folder: "b1g", Credentials: yandex.APIKey("k")}

	for name, cfg := range map[string]yandex.SpeechConfig{
		"адрес":   {Folder: full.Folder, Credentials: full.Credentials},
		"каталог": {Endpoint: full.Endpoint, Credentials: full.Credentials},
		"подпись": {Endpoint: full.Endpoint, Folder: full.Folder},
	} {
		if _, err := yandex.NewSpeech(cfg); err == nil {
			t.Errorf("%s: собрано", name)
		}
	}
}
