package gigachat_test

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"errors"
	"math/big"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/gigachat"
)

const key = "c2VjcmV0LWtleQ=="

// backend — OAuth и API на одном сервере: токены выдаются по порядку token-1, token-2, …
type backend struct {
	oauthCalls atomic.Int32
	oauth      func(w http.ResponseWriter, r *http.Request)
	api        func(w http.ResponseWriter, r *http.Request)
}

func (b *backend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/oauth" {
		n := b.oauthCalls.Add(1)
		if b.oauth != nil {
			b.oauth(w, r)

			return
		}

		issue(w, "token-"+strconv.Itoa(int(n)), time.Now().Add(30*time.Minute))

		return
	}

	if b.api != nil {
		b.api(w, r)
	}
}

func issue(w http.ResponseWriter, tok string, expires time.Time) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"access_token": tok, "expires_at": expires.UnixMilli()})
}

func config(url string) gigachat.Config {
	return gigachat.Config{
		OAuthEndpoint:    url + "/oauth",
		APIEndpoint:      url + "/api/",
		AuthorizationKey: key,
		Scope:            "GIGACHAT_API_PERS",
	}
}

func provider(t *testing.T, b *backend) *gigachat.Provider {
	t.Helper()

	srv := httptest.NewServer(b)
	t.Cleanup(srv.Close)

	p, err := gigachat.New(config(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	return p
}

func send(ctx context.Context, t *testing.T, p *gigachat.Provider, path string) error {
	t.Helper()

	resp, err := p.Send(ctx, path)
	if err != nil {
		return err
	}

	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("%s: HTTP %d", path, resp.StatusCode)
	}

	return nil
}

func bearer(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// TestOAuthRequestAndCache — Basic-ключ, RqUID v4 на каждый запрос, scope формой; токен переиспользуется.
func TestOAuthRequestAndCache(t *testing.T) {
	t.Parallel()

	uuid := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

	var (
		mu    sync.Mutex
		rqUID []string
	)

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}

		switch {
		case r.Method != http.MethodPost:
			t.Errorf("метод %s", r.Method)
		case r.Header.Get("Authorization") != "Basic "+key:
			t.Errorf("Authorization %q", r.Header.Get("Authorization"))
		case r.Header.Get("Content-Type") != "application/x-www-form-urlencoded":
			t.Errorf("Content-Type %q", r.Header.Get("Content-Type"))
		case r.PostForm.Get("scope") != "GIGACHAT_API_PERS":
			t.Errorf("scope %q", r.PostForm.Get("scope"))
		case !uuid.MatchString(r.Header.Get("Rquid")):
			t.Errorf("RqUID %q", r.Header.Get("Rquid"))
		}

		mu.Lock()
		rqUID = append(rqUID, r.Header.Get("Rquid"))
		mu.Unlock()

		issue(w, "token-"+strconv.Itoa(len(rqUID)), time.Now().Add(30*time.Minute))
	}
	b.api = func(_ http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models" || bearer(r) != "token-1" {
			t.Errorf("%s с токеном %q", r.URL.Path, bearer(r))
		}
	}

	p := provider(t, b)

	for range 3 {
		if err := send(context.Background(), t, p, "/models"); err != nil {
			t.Fatal(err)
		}
	}

	if n := b.oauthCalls.Load(); n != 1 {
		t.Errorf("запросов OAuth %d, ожидался один", n)
	}

	mu.Lock()
	defer mu.Unlock()

	if len(rqUID) != 1 {
		t.Fatalf("RqUID %v", rqUID)
	}
}

// TestTokenRefreshedWithMargin — токен обновляется до срока, а не после 401.
func TestTokenRefreshedWithMargin(t *testing.T) {
	t.Parallel()

	var clock atomic.Int64

	start := time.Now()
	clock.Store(start.UnixNano())

	var (
		mu   sync.Mutex
		seen []string
	)

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, _ *http.Request) {
		issue(w, "token-"+strconv.Itoa(int(b.oauthCalls.Load())), start.Add(30*time.Minute))
	}
	b.api = func(_ http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		seen = append(seen, bearer(r))
	}

	p := provider(t, b)
	p.SetClock(func() time.Time { return time.Unix(0, clock.Load()) })

	for _, at := range []time.Duration{0, 24 * time.Minute, 26 * time.Minute} {
		clock.Store(start.Add(at).UnixNano())

		if err := send(context.Background(), t, p, "/models"); err != nil {
			t.Fatal(err)
		}
	}

	mu.Lock()
	defer mu.Unlock()

	if want := []string{"token-1", "token-1", "token-2"}; strings.Join(seen, ",") != strings.Join(want, ",") {
		t.Errorf("токены %v, ожидались %v", seen, want)
	}
}

// TestFailedEarlyRefreshKeepsLiveToken — отказ обновления в запасе срока отдаёт ещё живой токен.
func TestFailedEarlyRefreshKeepsLiveToken(t *testing.T) {
	t.Parallel()

	var clock atomic.Int64

	start := time.Now()
	clock.Store(start.UnixNano())

	var seen atomic.Value

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, _ *http.Request) {
		if b.oauthCalls.Load() > 1 {
			w.WriteHeader(http.StatusBadGateway)

			return
		}

		issue(w, "token-1", start.Add(30*time.Minute))
	}
	b.api = func(_ http.ResponseWriter, r *http.Request) { seen.Store(bearer(r)) }

	p := provider(t, b)
	p.SetClock(func() time.Time { return time.Unix(0, clock.Load()) })

	for _, at := range []time.Duration{0, 26 * time.Minute} {
		clock.Store(start.Add(at).UnixNano())

		if err := send(context.Background(), t, p, "/models"); err != nil {
			t.Fatalf("%s: %v", at, err)
		}
	}

	if seen.Load() != "token-1" || b.oauthCalls.Load() != 2 {
		t.Errorf("токен %v, запросов OAuth %d", seen.Load(), b.oauthCalls.Load())
	}

	clock.Store(start.Add(31 * time.Minute).UnixNano())

	if err := send(context.Background(), t, p, "/models"); err == nil {
		t.Error("истёкший токен отдан после отказа обновления")
	}
}

// TestAbandonedRefreshDoesNotOverwrite — брошенное обновление, ответившее позже следующего,
// не подменяет его токен в кеше.
func TestAbandonedRefreshDoesNotOverwrite(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})

	var calls atomic.Int32

	cache := gigachat.NewTokenCache(func(context.Context) (string, time.Time, error) {
		if calls.Add(1) == 1 {
			<-release

			return "old", time.Now().Add(time.Hour), nil
		}

		return "new", time.Now().Add(time.Hour), nil
	})

	ctx, cancel := context.WithCancel(context.Background())
	abandoned := make(chan error)

	go func() {
		_, err := cache.Get(ctx)
		abandoned <- err
	}()

	var done <-chan struct{}

	for waiters := 0; waiters == 0; {
		time.Sleep(time.Millisecond)

		waiters, done = cache.Flight()
	}

	cancel()

	if err := <-abandoned; !errors.Is(err, context.Canceled) {
		t.Fatalf("брошенный вызов: %v", err)
	}

	if tok, err := cache.Get(context.Background()); err != nil || tok != "new" {
		t.Fatalf("новое обновление: %q, %v", tok, err)
	}

	close(release)
	<-done

	if tok, err := cache.Get(context.Background()); err != nil || tok != "new" {
		t.Errorf("после брошенного обновления: %q, %v", tok, err)
	}
}

// TestConcurrentRefreshMerged — десять вызовов на пустом кеше дают один запрос OAuth.
func TestConcurrentRefreshMerged(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, _ *http.Request) {
		<-release
		issue(w, "token-1", time.Now().Add(30*time.Minute))
	}

	p := provider(t, b)

	const callers = 10

	var wg sync.WaitGroup
	for range callers {
		wg.Go(func() {
			if err := send(context.Background(), t, p, "/models"); err != nil {
				t.Error(err)
			}
		})
	}

	waitFor(t, func() bool { return p.Waiters() == callers })
	close(release)
	wg.Wait()

	if n := b.oauthCalls.Load(); n != 1 {
		t.Errorf("запросов OAuth %d, ожидался один", n)
	}
}

// TestCancelledWaiterKeepsRefresh — отмена одного ждущего не отменяет обновление, нужное другому,
// а отмена последнего — отменяет.
func TestCancelledWaiterKeepsRefresh(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	aborted := make(chan struct{}, 1)

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
			issue(w, "token-"+strconv.Itoa(int(b.oauthCalls.Load())), time.Now().Add(30*time.Minute))
		case <-r.Context().Done():
		}
	}

	p := provider(t, b)

	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)

	go func() { first <- send(ctx, t, p, "/models") }()

	waitFor(t, func() bool { return p.Waiters() == 1 })

	second := make(chan error, 1)

	go func() { second <- send(context.Background(), t, p, "/models") }()

	waitFor(t, func() bool { return p.Waiters() == 2 })
	cancel()

	err := <-first

	var phased *llm.PhaseError
	if !errors.As(err, &phased) || phased.Phase != llm.PhaseAuth || !errors.Is(err, context.Canceled) {
		t.Errorf("отменённый ждущий: %v", err)
	}

	close(release)

	if err = <-second; err != nil {
		t.Errorf("второй ждущий: %v", err)
	}

	if n := b.oauthCalls.Load(); n != 1 {
		t.Errorf("запросов OAuth %d, ожидался один", n)
	}

	arrived := make(chan struct{}, 1)
	lone := &backend{oauth: func(_ http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm() // пока тело не дочитано, сервер не замечает обрыва соединения
		arrived <- struct{}{}
		<-r.Context().Done()
		aborted <- struct{}{}
	}}
	p2 := provider(t, lone)
	ctx2, cancel2 := context.WithCancel(context.Background())
	done := make(chan error, 1)

	go func() { done <- send(ctx2, t, p2, "/models") }()

	<-arrived
	cancel2()
	<-done

	select {
	case <-aborted:
	case <-time.After(5 * time.Second):
		t.Fatal("обновление без ждущих не отменено")
	}
}

// TestLate401WithOldToken — поздний 401 старым токеном не гасит уже обновлённый и не зовёт OAuth ещё раз.
func TestLate401WithOldToken(t *testing.T) {
	t.Parallel()

	slowArrived := make(chan struct{})
	fastDone := make(chan struct{})

	b := &backend{}
	b.api = func(w http.ResponseWriter, r *http.Request) {
		if bearer(r) != "token-1" {
			return
		}

		if r.URL.Path == "/api/slow" {
			close(slowArrived)
			<-fastDone
		}

		w.WriteHeader(http.StatusUnauthorized)
	}

	p := provider(t, b)
	slow := make(chan error, 1)

	go func() { slow <- send(context.Background(), t, p, "/slow") }()

	<-slowArrived

	if err := send(context.Background(), t, p, "/fast"); err != nil {
		t.Fatal(err)
	}

	close(fastDone)

	if err := <-slow; err != nil {
		t.Fatal(err)
	}

	if n := b.oauthCalls.Load(); n != 2 {
		t.Errorf("запросов OAuth %d, ожидалось два", n)
	}
}

// TestSecond401NeedsConfiguration — после принудительного обновления второй 401 — отказ фазы auth без цикла.
func TestSecond401NeedsConfiguration(t *testing.T) {
	t.Parallel()

	var apiCalls atomic.Int32

	b := &backend{}
	b.api = func(w http.ResponseWriter, _ *http.Request) {
		apiCalls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"status":401,"message":"Unauthorized"}`))
	}

	p := provider(t, b)
	_, err := p.Send(context.Background(), "/models")

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Status != http.StatusUnauthorized || status.Phase != llm.PhaseAuth ||
		status.Message != "Unauthorized" {
		t.Fatalf("отказ %v", err)
	}

	if b.oauthCalls.Load() != 2 || apiCalls.Load() != 2 {
		t.Errorf("OAuth %d, API %d, ожидалось по два", b.oauthCalls.Load(), apiCalls.Load())
	}
}

// TestOAuthRejected — отказ OAuth несёт фазу auth и текст конверта, но не ключ.
func TestOAuthRejected(t *testing.T) {
	t.Parallel()

	b := &backend{}
	b.oauth = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"code":6,"message":"credentials doesn't match db data"}`))
	}

	p := provider(t, b)
	_, err := p.Send(context.Background(), "/models")

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Status != http.StatusUnauthorized || status.Phase != llm.PhaseAuth {
		t.Fatalf("отказ %v", err)
	}

	if !strings.Contains(err.Error(), "credentials") || strings.Contains(err.Error(), key) {
		t.Errorf("текст отказа %q", err)
	}

	empty := &backend{
		oauth: func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"access_token":""}`)) },
	}

	_, err = provider(t, empty).Send(context.Background(), "/models")

	var (
		phased   *llm.PhaseError
		response *llm.ResponseError
	)
	if !errors.As(err, &phased) || phased.Phase != llm.PhaseAuth || !errors.As(err, &response) {
		t.Errorf("пустой токен: %v", err)
	}
}

// TestNewValidates — пустые поля отбиваются без сети и без ключа в тексте.
func TestNewValidates(t *testing.T) {
	t.Parallel()

	for name, mutate := range map[string]func(*gigachat.Config){
		"oauth": func(c *gigachat.Config) { c.OAuthEndpoint = "" },
		"api":   func(c *gigachat.Config) { c.APIEndpoint = "" },
		"key":   func(c *gigachat.Config) { c.AuthorizationKey = "" },
		"scope": func(c *gigachat.Config) { c.Scope = "" },
		"transport": func(c *gigachat.Config) {
			c.CA = x509.NewCertPool()
			c.HTTP = &http.Client{Transport: roundTrip(func(*http.Request) (*http.Response, error) { return nil, errors.New("не зовётся") })}
		},
	} {
		cfg := config("https://example.invalid")
		mutate(&cfg)

		if _, err := gigachat.New(cfg); err == nil || strings.Contains(err.Error(), key) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// TestCA — свой корень пропускает, чужой отбивает и тогда, когда основа транспорта выключила проверку.
func TestCA(t *testing.T) {
	t.Parallel()

	srv := httptest.NewTLSServer(&backend{})
	t.Cleanup(srv.Close)

	own := x509.NewCertPool()
	own.AddCert(srv.Certificate())

	cfg := config(srv.URL)
	cfg.CA = own

	p, err := gigachat.New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	if err = send(context.Background(), t, p, "/models"); err != nil {
		t.Fatalf("свой корень: %v", err)
	}

	insecure := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	cfg.CA = foreignPool(t)
	cfg.HTTP = &http.Client{Transport: insecure}

	if p, err = gigachat.New(cfg); err != nil {
		t.Fatal(err)
	}

	_, err = p.Send(context.Background(), "/models")

	var (
		unknown x509.UnknownAuthorityError
		phased  *llm.PhaseError
	)
	if !errors.As(err, &unknown) || !errors.As(err, &phased) || phased.Phase != llm.PhaseAuth {
		t.Errorf("чужой корень: %v", err)
	}

	if !insecure.TLSClientConfig.InsecureSkipVerify {
		t.Error("транспорт основы изменён")
	}
}

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// foreignPool — пул с самоподписанным корнем, которым сервер не подписан.
func foreignPool(t *testing.T) *x509.CertPool {
	t.Helper()

	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "foreign root"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		t.Fatal(err)
	}

	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	pool := x509.NewCertPool()
	pool.AddCert(cert)

	return pool
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()

	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("условие не наступило")
		}

		time.Sleep(time.Millisecond)
	}
}
