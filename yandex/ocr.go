package yandex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// OCRName — имя плеча распознавания текста в отчётах и журнале.
const OCRName = "visionocr"

const (
	// ocrMaxBytes — предел файла у `recognizeText` любой модели.
	ocrMaxBytes = 10 << 20
	// maxOCRBody — `textAnnotation` несёт блоки, строки и слова с координатами: плотная страница — мегабайты.
	maxOCRBody = 8 << 20
)

// OCRConfig — адрес, каталог и подпись Vision OCR.
type OCRConfig struct {
	// Endpoint — корень API v1 до `/recognizeText`: другой хост, чем у чата и речи.
	Endpoint    string
	Folder      string
	Credentials CredentialSource
	HTTP        *http.Client
}

// OCR — синхронное распознавание Vision OCR; состояния нет.
type OCR struct {
	endpoint string
	folder   string
	creds    CredentialSource
	http     *http.Client
}

// NewOCR собирает плечо без сети.
func NewOCR(cfg OCRConfig) (*OCR, error) {
	switch {
	case cfg.Endpoint == "":
		return nil, errors.New("visionocr: адрес API не задан")
	case cfg.Folder == "":
		return nil, errors.New("visionocr: каталог не задан")
	case cfg.Credentials == nil:
		return nil, errors.New("visionocr: подпись не задана")
	}

	client := cfg.HTTP
	if client == nil {
		client = http.DefaultClient
	}

	return &OCR{
		endpoint: strings.TrimRight(cfg.Endpoint, "/"),
		folder:   cfg.Folder,
		creds:    cfg.Credentials,
		http:     client,
	}, nil
}

// Name — [OCRName].
func (o *OCR) Name() string { return OCRName }

// OCRCapabilities — известна только модель `page`.
func (o *OCR) OCRCapabilities(model string) (llm.OCRCapabilities, bool) {
	if model == "page" {
		return llm.OCRCapabilities{MaxBytes: ocrMaxBytes}, true
	}

	return llm.OCRCapabilities{}, false
}

type ocrRequest struct {
	MIMEType      string   `json:"mimeType"`
	LanguageCodes []string `json:"languageCodes,omitempty"`
	Model         string   `json:"model"`
	Content       string   `json:"content"`
}

// Recognize — одна попытка. Предел файла сверяется и у неизвестной модели: он у API, а не у модели.
func (o *OCR) Recognize(ctx context.Context, req llm.OCRRequest) (llm.OCRText, error) {
	if err := req.Validate(ocrMaxBytes); err != nil {
		return llm.OCRText{}, err
	}

	scheme, value, err := o.creds.Token(ctx)
	if err != nil {
		return llm.OCRText{}, &llm.PhaseError{Phase: llm.PhaseAuth, Err: err}
	}

	mimeType := "JPEG"
	if req.MIME == llm.MIMEPNG {
		mimeType = "PNG"
	}

	body, err := json.Marshal(ocrRequest{
		MIMEType:      mimeType,
		LanguageCodes: req.Languages,
		Model:         req.Model,
		Content:       base64.StdEncoding.EncodeToString(req.Image),
	})
	if err != nil {
		return llm.OCRText{}, &llm.RequestError{Message: "тело visionocr", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.endpoint+"/recognizeText", bytes.NewReader(body))
	if err != nil {
		return llm.OCRText{}, &llm.RequestError{Message: "запрос visionocr", Err: err}
	}

	httpReq.Header.Set("Authorization", scheme+" "+value)
	httpReq.Header.Set("X-Folder-Id", o.folder)
	httpReq.Header.Set("X-Data-Logging-Enabled", "false")
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	resp, err := o.http.Do(httpReq)
	if err != nil {
		return llm.OCRText{}, fmt.Errorf("запрос /recognizeText: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, ocrErrorMessage)

		return llm.OCRText{}, &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			RequestID:  requestID,
		}
	}

	var env struct {
		Result struct {
			TextAnnotation struct {
				FullText string `json:"fullText"`
			} `json:"textAnnotation"`
		} `json:"result"`
	}
	if err = httpjson.Decode(resp.Body, maxOCRBody, &env); err != nil {
		return llm.OCRText{}, &llm.ResponseError{Message: "ответ /recognizeText", RequestID: requestID, Err: err}
	}

	return llm.OCRText{
		Text:      strings.TrimSpace(env.Result.TextAnnotation.FullText),
		Model:     req.Model,
		RequestID: requestID,
	}, nil
}

// ocrErrorMessage — текст конверта Vision `{"code", "message"}`; code у шлюза gRPC — число.
func ocrErrorMessage(raw []byte) string {
	var body struct {
		Code    json.RawMessage `json:"code"`
		Message string          `json:"message"`
	}

	_ = json.Unmarshal(raw, &body)

	if body.Message == "" {
		return strings.Trim(string(body.Code), `"`)
	}

	return body.Message
}

var _ llm.Recognizer = (*OCR)(nil)
