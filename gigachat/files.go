package gigachat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// Пределы кадров GigaChat; профиль возможностей объявляет те же.
const (
	MaxImagesPerMessage = 1
	MaxImagesPerRequest = 10
	// MaxImageBytes — предел файла изображения у `/files`.
	MaxImageBytes = 15 << 20
)

// CleanupBudget — срок уборки файлов после попытки, отдельный от её срока: входит в бюджет вызова сверху.
const CleanupBudget = 10 * time.Second

// maxFileBody — предел ответа `/files`: описание одного файла.
const maxFileBody = 64 << 10

// imageExt — расширение имени файла по MIME; пустое — тип GigaChat не принимает.
func imageExt(mime string) string {
	switch strings.ToLower(strings.TrimSpace(mime)) {
	case "image/jpeg", "image/jpg":
		return "jpg"
	case "image/png":
		return "png"
	case "image/tiff":
		return "tiff"
	case "image/bmp":
		return "bmp"
	default:
		return ""
	}
}

// checkImages отбивает кадры, которые загрузка не примет, до первого байта в сеть.
func checkImages(msgs []llm.Message) error {
	total := 0

	for i, m := range msgs {
		if len(m.Images) > MaxImagesPerMessage {
			return fmt.Errorf("сообщение %d: кадров больше %d", i, MaxImagesPerMessage)
		}

		for j, img := range m.Images {
			switch {
			case imageExt(img.MIME) == "":
				return fmt.Errorf("сообщение %d, кадр %d: тип %q не принимается", i, j, img.MIME)
			case len(img.Data) == 0:
				return fmt.Errorf("сообщение %d, кадр %d: пустой", i, j)
			case len(img.Data) > MaxImageBytes:
				return fmt.Errorf("сообщение %d, кадр %d: больше %d байт", i, j, MaxImageBytes)
			}
		}

		total += len(m.Images)
	}

	if total > MaxImagesPerRequest {
		return fmt.Errorf("кадров в запросе больше %d", MaxImagesPerRequest)
	}

	return nil
}

// withFiles загружает кадры, зовёт send с id файлов по сообщениям и убирает загруженное после попытки —
// удачной, отказавшей и прерванной на середине загрузки. Неудавшаяся уборка — предупреждение, не отказ.
func (p *Provider) withFiles(
	ctx context.Context,
	msgs []llm.Message,
	send func(ctx context.Context, attachments [][]string) (llm.Result, error),
) (llm.Result, error) {
	if err := checkImages(msgs); err != nil {
		return llm.Result{}, &llm.PhaseError{
			Phase: llm.PhaseUpload,
			Err:   &llm.RequestError{Message: "кадры gigachat", Err: err},
		}
	}

	attachments := make([][]string, len(msgs))

	var created []string

	var (
		res llm.Result
		err error
	)

upload:
	for i, m := range msgs {
		for j, img := range m.Images {
			var id string

			id, err = p.upload(ctx, img, "image-"+strconv.Itoa(i)+"-"+strconv.Itoa(j))
			if err != nil {
				break upload
			}

			created = append(created, id)
			attachments[i] = append(attachments[i], id)
		}
	}

	if err == nil {
		res, err = send(ctx, attachments)
	}

	warning := p.cleanup(ctx, created)

	switch {
	case warning == nil:
		return res, err
	case err != nil:
		return llm.Result{}, &llm.WarnedError{Err: err, Cleanup: warning}
	default:
		res.Cleanup = warning

		return res, nil
	}
}

// upload — один кадр в `/files` с purpose=general; отказы помечены фазой upload, если фазы ещё нет.
func (p *Provider) upload(ctx context.Context, img llm.Image, name string) (string, error) {
	var body bytes.Buffer

	form := multipart.NewWriter(&body)

	head := textproto.MIMEHeader{}
	head.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s.%s"`, name, imageExt(img.MIME)))
	head.Set("Content-Type", strings.ToLower(strings.TrimSpace(img.MIME)))

	part, err := form.CreatePart(head)
	if err == nil {
		_, err = part.Write(img.Data)
	}

	if err == nil {
		err = form.WriteField("purpose", "general")
	}

	if err == nil {
		err = form.Close()
	}

	if err != nil {
		return "", &llm.PhaseError{Phase: llm.PhaseUpload, Err: &llm.RequestError{Message: "форма файла", Err: err}}
	}

	resp, err := p.do(ctx, func(ctx context.Context) (*http.Request, error) {
		req, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, p.api+"/files", bytes.NewReader(body.Bytes()))
		if reqErr != nil {
			return nil, reqErr
		}

		req.Header.Set("Content-Type", form.FormDataContentType())
		req.Header.Set("Accept", "application/json")

		return req, nil
	})
	if err != nil {
		return "", inPhase(llm.PhaseUpload, err)
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, errorMessage)

		return "", &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			Phase:      llm.PhaseUpload,
			RequestID:  requestID,
		}
	}

	var file struct {
		ID string `json:"id"`
	}

	if err = httpjson.Decode(resp.Body, maxFileBody, &file); err == nil && file.ID == "" {
		err = errors.New("ответ без id файла")
	}

	if err != nil {
		return "", &llm.PhaseError{
			Phase: llm.PhaseUpload,
			Err:   &llm.ResponseError{Message: "ответ /files", RequestID: requestID, Err: err},
		}
	}

	return file.ID, nil
}

// cleanup удаляет файлы попытки под своим сроком: отменённая или просроченная попытка тоже убирает за собой.
func (p *Provider) cleanup(ctx context.Context, ids []string) *llm.CleanupWarning {
	if len(ids) == 0 {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), CleanupBudget)
	defer cancel()

	var (
		left []string
		errs []error
	)

	for _, id := range ids {
		if err := p.remove(ctx, id); err != nil {
			left = append(left, id)
			errs = append(errs, fmt.Errorf("файл %s: %w", id, err))
		}
	}

	if len(left) == 0 {
		return nil
	}

	return &llm.CleanupWarning{Files: left, Err: errors.Join(errs...)}
}

// remove — `POST /files/{id}/delete`; ответ без deleted=true — отказ.
func (p *Provider) remove(ctx context.Context, id string) error {
	resp, err := p.do(ctx, func(ctx context.Context) (*http.Request, error) {
		endpoint := p.api + "/files/" + url.PathEscape(id) + "/delete"

		req, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
		if reqErr != nil {
			return nil, reqErr
		}

		req.Header.Set("Accept", "application/json")

		return req, nil
	})
	if err != nil {
		return err
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, errorMessage)

		return &llm.StatusError{Status: st.Code, Message: st.Message, RequestID: resp.Header.Get("X-Request-Id")}
	}

	var out struct {
		Deleted bool `json:"deleted"`
	}

	if err = httpjson.Decode(resp.Body, maxFileBody, &out); err != nil {
		return err
	}

	if !out.Deleted {
		return errors.New("поставщик не подтвердил удаление")
	}

	return nil
}

// inPhase помечает отказ фазой, если его ещё не пометили: отказ входа внутри загрузки остаётся отказом входа.
func inPhase(phase llm.Phase, err error) error {
	var phased *llm.PhaseError
	if errors.As(err, &phased) && phased.Phase != "" {
		return err
	}

	var status *llm.StatusError
	if errors.As(err, &status) && status.Phase != "" {
		return err
	}

	return &llm.PhaseError{Phase: phase, Err: err}
}
