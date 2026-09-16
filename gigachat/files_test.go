package gigachat_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/gigachat"
)

// files — `/files` на сервере-двойнике: id выдаются по порядку f1, f2, …; отказ загрузки и удаления настраивается.
type files struct {
	mu       sync.Mutex
	uploads  []upload
	deleted  []string
	failNth  int // номер загрузки, отвечающей 500; ноль — все удаются
	brokeNth int // номер загрузки, принятой с оборванным ответом
	failFrom string
}

type upload struct {
	mime, purpose, token string
	data                 []byte
}

func (f *files) serve(t *testing.T) func(w http.ResponseWriter, r *http.Request) {
	t.Helper()

	return func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()

		if r.Method != http.MethodPost {
			t.Errorf("метод %s у %s", r.Method, r.URL.Path)
		}

		switch {
		case r.URL.Path == "/api/files":
			f.upload(t, w, r)
		case strings.HasPrefix(r.URL.Path, "/api/files/") && strings.HasSuffix(r.URL.Path, "/delete"):
			id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/files/"), "/delete")
			if f.failFrom != "" && id >= f.failFrom {
				w.WriteHeader(http.StatusInternalServerError)

				return
			}

			f.deleted = append(f.deleted, id)
			_ = json.NewEncoder(w).Encode(map[string]any{"id": id, "deleted": true})
		default:
			t.Errorf("неожиданный путь %s", r.URL.Path)
		}
	}
}

func (f *files) upload(t *testing.T, w http.ResponseWriter, r *http.Request) {
	t.Helper()

	file, head, err := r.FormFile("file")
	if err != nil {
		t.Error(err)

		return
	}

	data, _ := io.ReadAll(file)
	f.uploads = append(f.uploads, upload{
		mime: head.Header.Get("Content-Type"), purpose: r.FormValue("purpose"), token: bearer(r), data: data,
	})

	n := len(f.uploads)
	if n == f.failNth {
		w.Header().Set("X-Request-Id", "req-fail")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"message":"storage"}`))

		return
	}

	if n == f.brokeNth {
		_, _ = w.Write([]byte(`{"id":"f` + strconv.Itoa(n)))

		return
	}

	_ = json.NewEncoder(w).Encode(map[string]any{"id": "f" + strconv.Itoa(n), "object": "file", "purpose": "general"})
}

func (f *files) state() (int, []string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	return len(f.uploads), append([]string(nil), f.deleted...)
}

func jpeg(tag string) llm.Image { return llm.Image{MIME: "image/jpeg", Data: []byte("jpeg-" + tag)} }

// TestFilesUploadedAttachedAndRemoved — кадр уходит формой с purpose=general, id — в своё сообщение,
// после попытки загруженное удаляется.
func TestFilesUploadedAttachedAndRemoved(t *testing.T) {
	t.Parallel()

	f := &files{}
	p := provider(t, &backend{api: f.serve(t)})
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Text: "правила"},
		{Role: llm.RoleUser, Text: "кадр", Images: []llm.Image{jpeg("a")}},
		{Role: llm.RoleUser, Text: "ещё", Images: []llm.Image{{MIME: "image/PNG", Data: []byte("png")}}},
	}

	var got [][]string

	res, err := p.WithFiles(context.Background(), msgs, func(_ context.Context, att [][]string) (llm.Result, error) {
		got = att

		return llm.Result{Text: "ok"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if want := [][]string{nil, {"f1"}, {"f2"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("attachments %v, ожидалось %v", got, want)
	}

	if res.Text != "ok" || res.Cleanup != nil {
		t.Errorf("результат %+v", res)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	first, second := f.uploads[0], f.uploads[1]
	if first.mime != "image/jpeg" || first.purpose != "general" || first.token != "token-1" ||
		!bytes.Equal(first.data, []byte("jpeg-a")) || second.mime != "image/png" {
		t.Errorf("загрузки %+v", f.uploads)
	}

	if !reflect.DeepEqual(f.deleted, []string{"f1", "f2"}) {
		t.Errorf("удалены %v", f.deleted)
	}
}

// TestPartialUploadRemoved — отказ второй загрузки: генерации нет, первый файл удалён, отказ фазы upload.
func TestPartialUploadRemoved(t *testing.T) {
	t.Parallel()

	f := &files{failNth: 2}
	p := provider(t, &backend{api: f.serve(t)})
	msgs := []llm.Message{
		{Role: llm.RoleUser, Images: []llm.Image{jpeg("a")}},
		{Role: llm.RoleUser, Images: []llm.Image{jpeg("b")}},
		{Role: llm.RoleUser, Images: []llm.Image{jpeg("c")}},
	}

	sent := 0

	_, err := p.WithFiles(context.Background(), msgs, func(context.Context, [][]string) (llm.Result, error) {
		sent++

		return llm.Result{}, nil
	})

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Status != http.StatusInternalServerError ||
		status.Phase != llm.PhaseUpload || status.RequestID != "req-fail" || status.Message != "storage" {
		t.Fatalf("отказ %v (%+v)", err, status)
	}

	uploads, deleted := f.state()
	if sent != 0 || uploads != 2 || !reflect.DeepEqual(deleted, []string{"f1"}) {
		t.Errorf("генераций %d, загрузок %d, удалены %v", sent, uploads, deleted)
	}

	var warned *llm.WarnedError
	if errors.As(err, &warned) {
		t.Errorf("отказ статусом файла не оставляет: %+v", warned.Cleanup)
	}
}

// TestUnansweredUploadIsUncertain — принятый сервером файл с оборванным ответом неубираем,
// и предупреждение уборки говорит об этом, хотя известные файлы убраны.
func TestUnansweredUploadIsUncertain(t *testing.T) {
	t.Parallel()

	f := &files{brokeNth: 2}
	p := provider(t, &backend{api: f.serve(t)})
	msgs := []llm.Message{
		{Role: llm.RoleUser, Images: []llm.Image{jpeg("a")}},
		{Role: llm.RoleUser, Images: []llm.Image{jpeg("b")}},
	}

	_, err := p.WithFiles(context.Background(), msgs, func(context.Context, [][]string) (llm.Result, error) {
		return llm.Result{}, nil
	})

	var warned *llm.WarnedError
	if !errors.As(err, &warned) || warned.Cleanup.Uncertain != 1 || len(warned.Cleanup.Files) != 0 ||
		warned.Cleanup.Err != nil {
		t.Fatalf("отказ %v", err)
	}

	if _, deleted := f.state(); !reflect.DeepEqual(deleted, []string{"f1"}) {
		t.Errorf("удалены %v", deleted)
	}
}

// TestCleanupAfterCancelledAttempt — отменённая попытка убирает за собой под своим сроком.
func TestCleanupAfterCancelledAttempt(t *testing.T) {
	t.Parallel()

	f := &files{}
	p := provider(t, &backend{api: f.serve(t)})
	ctx, cancel := context.WithCancel(context.Background())

	_, err := p.WithFiles(ctx, []llm.Message{{Role: llm.RoleUser, Images: []llm.Image{jpeg("a")}}},
		func(ctx context.Context, _ [][]string) (llm.Result, error) {
			cancel()

			return llm.Result{}, ctx.Err()
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("отказ %v", err)
	}

	if _, deleted := f.state(); !reflect.DeepEqual(deleted, []string{"f1"}) {
		t.Errorf("удалены %v", deleted)
	}
}

// generating — плечо поверх WithFiles с генерацией по сценарию; считает генерации.
type generating struct {
	p     *gigachat.Provider
	mu    sync.Mutex
	calls int
	err   error
}

func (g *generating) Name() string { return gigachat.Name }

func (g *generating) Capabilities(string) (llm.Capabilities, bool) {
	return llm.Capabilities{
		Vision:              true,
		MaxImagesPerMessage: gigachat.MaxImagesPerMessage,
		MaxImagesPerRequest: gigachat.MaxImagesPerRequest,
	}, true
}

func (g *generating) Complete(ctx context.Context, req llm.Request) (llm.Result, error) {
	return g.p.WithFiles(ctx, req.Messages, func(context.Context, [][]string) (llm.Result, error) {
		g.mu.Lock()
		defer g.mu.Unlock()

		g.calls++

		return llm.Result{Text: "ok", Usage: llm.Usage{Known: true}}, g.err
	})
}

// TestCleanupFailureWithoutRegeneration — неудавшееся удаление не повторяет оплаченную генерацию:
// ни у удачи, ни у отказа; предупреждение с оставшимися файлами едет в отчёт попытки.
func TestCleanupFailureWithoutRegeneration(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		err  error
	}{
		{name: "удача"},
		{name: "отказ без повтора", err: &llm.StatusError{Status: http.StatusBadRequest}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			f := &files{failFrom: "f2"}
			g := &generating{p: provider(t, &backend{api: f.serve(t)}), err: tc.err}
			c := llm.New(g, 5*time.Second)
			req := llm.Request{Model: "GigaChat-2-Max", Messages: []llm.Message{
				{Role: llm.RoleUser, Images: []llm.Image{jpeg("a")}},
				{Role: llm.RoleUser, Images: []llm.Image{jpeg("b")}},
			}}

			res, err := c.Chat(context.Background(), req)

			attempts := res.Attempts

			if tc.err != nil {
				var call *llm.CallError
				if !errors.As(err, &call) || call.Status != http.StatusBadRequest {
					t.Fatalf("отказ %v", err)
				}

				attempts = call.Attempts
			} else if err != nil || res.Cleanup == nil {
				t.Fatalf("результат %+v, отказ %v", res, err)
			}

			if g.calls != 1 || len(attempts) != 1 || attempts[0].Cleanup == nil ||
				!reflect.DeepEqual(attempts[0].Cleanup.Files, []string{"f2"}) || attempts[0].Cleanup.Err == nil {
				t.Errorf("генераций %d, попытки %+v", g.calls, attempts)
			}

			if _, deleted := f.state(); !reflect.DeepEqual(deleted, []string{"f1"}) {
				t.Errorf("удалены %v", deleted)
			}
		})
	}
}

// TestImageLimits — кадры сверх пределов GigaChat отбиваются до сети отказом без повтора;
// ровно десять по одному на сообщение проходят.
func TestImageLimits(t *testing.T) {
	t.Parallel()

	one := func(img llm.Image) []llm.Message {
		return []llm.Message{{Role: llm.RoleUser, Images: []llm.Image{img}}}
	}

	spread := func(n int) []llm.Message {
		msgs := make([]llm.Message, n)
		for i := range msgs {
			msgs[i] = llm.Message{Role: llm.RoleUser, Images: []llm.Image{jpeg(strconv.Itoa(i))}}
		}

		return msgs
	}

	for _, tc := range []struct {
		name string
		msgs []llm.Message
		ok   bool
	}{
		{name: "два в сообщении", msgs: []llm.Message{{Role: llm.RoleUser, Images: []llm.Image{jpeg("a"), jpeg("b")}}}},
		{name: "одиннадцать в запросе", msgs: spread(11)},
		{name: "десять в запросе", msgs: spread(10), ok: true},
		{name: "webp", msgs: one(llm.Image{MIME: "image/webp", Data: []byte("x")})},
		{name: "пустой", msgs: one(llm.Image{MIME: "image/jpeg"})},
		{name: "больше предела", msgs: one(llm.Image{MIME: "image/png", Data: make([]byte, gigachat.MaxImageBytes+1)})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			f := &files{}
			b := &backend{api: f.serve(t)}
			p := provider(t, b)

			_, err := p.WithFiles(context.Background(), tc.msgs, func(context.Context, [][]string) (llm.Result, error) {
				return llm.Result{}, nil
			})

			uploads, deleted := f.state()

			if tc.ok {
				if err != nil || uploads != 10 || len(deleted) != 10 {
					t.Errorf("отказ %v, загрузок %d, удалено %d", err, uploads, len(deleted))
				}

				return
			}

			var request *llm.RequestError

			var phased *llm.PhaseError

			if !errors.As(err, &request) || !errors.As(err, &phased) || phased.Phase != llm.PhaseUpload {
				t.Errorf("отказ %v", err)
			}

			if uploads != 0 || b.oauthCalls.Load() != 0 {
				t.Errorf("до сети дошло: загрузок %d, OAuth %d", uploads, b.oauthCalls.Load())
			}
		})
	}
}
