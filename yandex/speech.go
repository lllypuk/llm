package yandex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// SpeechName — имя плеча распознавания в отчётах и журнале.
const SpeechName = "speechkit"

const (
	// speechMaxAudio — предел синхронного `stt:recognize` у любой модели; 1 МБ при 16 кГц он не превышает.
	speechMaxAudio = 30 * time.Second
	maxSpeechBody  = 64 << 10
)

// SpeechConfig — адрес, каталог и подпись SpeechKit.
type SpeechConfig struct {
	// Endpoint — корень API v1 до `/stt:recognize`: другой хост, чем у чатового [Config.Endpoint].
	Endpoint    string
	Folder      string
	Credentials CredentialSource
	HTTP        *http.Client
}

// Speech — синхронное распознавание SpeechKit v1; состояния нет.
type Speech struct {
	endpoint string
	folder   string
	creds    CredentialSource
	http     *http.Client
}

// NewSpeech собирает плечо без сети.
func NewSpeech(cfg SpeechConfig) (*Speech, error) {
	switch {
	case cfg.Endpoint == "":
		return nil, errors.New("speechkit: адрес API не задан")
	case cfg.Folder == "":
		return nil, errors.New("speechkit: каталог не задан")
	case cfg.Credentials == nil:
		return nil, errors.New("speechkit: подпись не задана")
	}

	client := cfg.HTTP
	if client == nil {
		client = http.DefaultClient
	}

	return &Speech{
		endpoint: strings.TrimRight(cfg.Endpoint, "/"),
		folder:   cfg.Folder,
		creds:    cfg.Credentials,
		http:     client,
	}, nil
}

// Name — [SpeechName].
func (s *Speech) Name() string { return SpeechName }

// SpeechCapabilities — модель здесь — `topic` API; неизвестная не подтверждается.
func (s *Speech) SpeechCapabilities(model string) (llm.SpeechCapabilities, bool) {
	switch model {
	case "general", "general:rc", "general:deprecated":
		return llm.SpeechCapabilities{MaxAudio: speechMaxAudio}, true
	}

	return llm.SpeechCapabilities{}, false
}

// Transcribe — одна попытка. Предел длительности сверяется и у неизвестной модели: он у API, а не у модели.
func (s *Speech) Transcribe(ctx context.Context, req llm.SpeechRequest) (llm.Transcript, error) {
	if err := req.Validate(speechMaxAudio); err != nil {
		return llm.Transcript{}, err
	}

	scheme, value, err := s.creds.Token(ctx)
	if err != nil {
		return llm.Transcript{}, &llm.PhaseError{Phase: llm.PhaseAuth, Err: err}
	}

	query := url.Values{
		"topic":           {req.Model},
		"format":          {"lpcm"},
		"sampleRateHertz": {strconv.Itoa(req.SampleRate)},
		"folderId":        {s.folder},
	}
	if req.Language != "" {
		query.Set("lang", req.Language)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.endpoint+"/stt:recognize?"+query.Encode(), bytes.NewReader(req.PCM))
	if err != nil {
		return llm.Transcript{}, &llm.RequestError{Message: "запрос speechkit", Err: err}
	}

	httpReq.Header.Set("Authorization", scheme+" "+value)
	httpReq.Header.Set("Content-Type", "application/octet-stream")
	httpReq.Header.Set("Accept", "application/json")

	resp, err := s.http.Do(httpReq)
	if err != nil {
		return llm.Transcript{}, fmt.Errorf("запрос /stt:recognize: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, speechErrorMessage)

		return llm.Transcript{}, &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			RequestID:  requestID,
		}
	}

	var env struct {
		Result string `json:"result"`
	}
	if err = httpjson.Decode(resp.Body, maxSpeechBody, &env); err != nil {
		return llm.Transcript{}, &llm.ResponseError{Message: "ответ /stt:recognize", RequestID: requestID, Err: err}
	}

	return llm.Transcript{
		Text:        strings.TrimSpace(env.Result),
		Model:       req.Model,
		RequestID:   requestID,
		AudioMillis: req.Duration().Milliseconds(),
	}, nil
}

// speechErrorMessage — текст конверта SpeechKit `{"error_code", "error_message"}`.
func speechErrorMessage(raw []byte) string {
	var body struct {
		Code    string `json:"error_code"`
		Message string `json:"error_message"`
	}

	_ = json.Unmarshal(raw, &body)

	if body.Message == "" {
		return body.Code
	}

	return body.Message
}

var _ llm.Transcriber = (*Speech)(nil)
