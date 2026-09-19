//go:build live

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const (
	testRate    = 16000
	spokenText  = "проверка связи"
	testAPIKey  = "speech-key"
	secondClip  = 2 * testRate
	silentClip  = testRate / 2
	speechTimer = time.Minute
)

// wav собирает запись с посторонним чанком LIST перед data: разборщик обязан его пропустить.
func wav(format, channels, bits uint16, pcm []byte) []byte {
	var b bytes.Buffer

	chunk := func(id string, body []byte) {
		b.WriteString(id)
		_ = binary.Write(&b, binary.LittleEndian, uint32(len(body)))
		b.Write(body)

		if len(body)%2 == 1 {
			b.WriteByte(0)
		}
	}

	fmtBody := make([]byte, fmtMinSize)
	binary.LittleEndian.PutUint16(fmtBody[0:], format)
	binary.LittleEndian.PutUint16(fmtBody[2:], channels)
	binary.LittleEndian.PutUint32(fmtBody[4:], testRate)
	binary.LittleEndian.PutUint32(fmtBody[8:], testRate*uint32(channels)*uint32(bits)/8)
	binary.LittleEndian.PutUint16(fmtBody[12:], channels*bits/8)
	binary.LittleEndian.PutUint16(fmtBody[14:], bits)

	b.WriteString("RIFF\x00\x00\x00\x00WAVE")
	chunk("fmt ", fmtBody)
	chunk("LIST", []byte("odd"))
	chunk("data", pcm)

	out := b.Bytes()
	binary.LittleEndian.PutUint32(out[4:], uint32(len(out)-8))

	return out
}

// tone — сэмплы с ненулевым значением: фейковое плечо отвечает на них текстом, на тишину — пустотой.
func tone(samples int) []byte { return bytes.Repeat([]byte{0x10, 0x00}, samples) }

func goodWAV(pcm []byte) []byte { return wav(formatPCM, monoChannels, bitsPerSample, pcm) }

func TestParseWAV(t *testing.T) {
	t.Parallel()

	c, err := parseWAV(goodWAV(tone(testRate)))
	if err != nil || c.rate != testRate || len(c.pcm) != 2*testRate {
		t.Fatalf("частота %d, байт %d, ошибка %v", c.rate, len(c.pcm), err)
	}

	good := goodWAV(tone(testRate))
	noData := good[:len(good)-2*testRate-chunkHeader]
	binary.LittleEndian.PutUint32(noData[4:], uint32(len(noData)-8))

	cases := map[string]struct {
		raw  []byte
		want string
	}{
		"не RIFF":       {[]byte("ID3\x04not a wave file"), "не RIFF/WAVE"},
		"пусто":         {nil, "не RIFF/WAVE"},
		"не PCM":        {wav(3, monoChannels, 32, tone(2)), "нужен PCM"},
		"стерео":        {wav(formatPCM, 2, bitsPerSample, tone(2)), "каналов 2"},
		"8 бит":         {wav(formatPCM, monoChannels, 8, tone(2)), "8 бит"},
		"усечённый":     {good[:len(good)-10], `"data" усечён`},
		"без data":      {noData, "нет чанка data"},
		"data без fmt":  {[]byte("RIFF\x0c\x00\x00\x00WAVEdata\x00\x00\x00\x00"), "раньше fmt"},
		"короткий fmt ": {[]byte("RIFF\x10\x00\x00\x00WAVEfmt \x02\x00\x00\x00\x01\x00"), "короче 16"},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if _, parseErr := parseWAV(tc.raw); parseErr == nil || !strings.Contains(parseErr.Error(), tc.want) {
				t.Fatalf("ошибка %v, ждали %q", parseErr, tc.want)
			}
		})
	}
}

// fakeSpeechKit — синхронный stt:recognize: на тишину пустой результат, на звук — spokenText.
func fakeSpeechKit(t *testing.T, calls *atomic.Int32) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)

		if r.Header.Get("Authorization") != "Api-Key "+testAPIKey ||
			r.URL.Query().Get("sampleRateHertz") != "16000" {
			w.WriteHeader(http.StatusBadRequest)

			return
		}

		body, _ := io.ReadAll(r.Body)
		text := ""

		if bytes.ContainsFunc(body, func(r rune) bool { return r != 0 }) {
			text = spokenText
		}

		_ = json.NewEncoder(w).Encode(map[string]string{"result": text})
	}))
	t.Cleanup(srv.Close)

	return srv
}

func writeSpeechConfig(t *testing.T, endpoint string) string {
	t.Helper()

	return writeFile(t, `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "http://ollama.invalid"},
    "stt": {"kind": "speechkit", "endpoint": "`+endpoint+`", "folder": "b1g", "auth": {"api_key": "${STT_KEY}"}},
    "salute": {"kind": "salutespeech", "endpoint": "https://salute.invalid", "oauth_endpoint": "https://oauth.invalid",
               "scope": "SALUTE_SPEECH_PERS", "auth": {"authorization_key": "${SALUTE_KEY_UNSET}"}}
  },
  "tasks": {"ask": {"provider": "local", "model": "gemma"}},
  "speech": {
    "dictation": {"provider": "stt", "model": "general", "language": "ru-RU", "attempts": 1, "price_plan": "stt"},
    "backup": {"provider": "salute", "model": "general"}
  },
  "prices": {
    "stt": [{"revision": "stt-1", "currency": "RUB", "valid_from": "2026-01-01T00:00:00Z",
             "audio_step": "15s", "rates": {"audio_second": 10667}}]
  }
}`)
}

func speechEnv(name string) (string, bool) {
	if name == "STT_KEY" {
		return testAPIKey, true
	}

	return "", false
}

func writeClips(t *testing.T, clips map[string][]byte) string {
	t.Helper()

	dir := t.TempDir()
	for name, raw := range clips {
		if err := os.WriteFile(filepath.Join(dir, name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	return dir
}

func speechOpts(config, dir string) speechOptions {
	return speechOptions{
		config:      config,
		dir:         dir,
		task:        "dictation",
		maxFiles:    defaultMaxFiles,
		maxRequests: defaultMaxRequests,
		maxCost:     defaultMaxCost,
		timeout:     speechTimer,
	}
}

func TestSpeechRun(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeSpeechKit(t, &calls)
	dir := writeClips(t, map[string][]byte{
		"a.wav":     goodWAV(tone(secondClip)),
		"b.WAV":     goodWAV(make([]byte, 2*silentClip)),
		"notes.txt": []byte("не запись"),
	})

	var stdout, stderr bytes.Buffer

	code := runSpeech(context.Background(), speechOpts(writeSpeechConfig(t, srv.URL), dir), &stdout, &stderr, speechEnv)
	if code != exitOK {
		t.Fatalf("код %d, stderr %s\n%s", code, stderr.String(), stdout.String())
	}

	out := stdout.String()
	for _, want := range []string{
		"- задача: dictation: stt/general, язык ru-RU",
		"- [pass] speech a.wav: задержка", "секунд 2.00, попыток 1, стоимость 0.160005 RUB по stt-1 (estimated)",
		"- [pass] speech b.WAV", "секунд 0.50", "пустой текст",
		"пройдено 2, нарушений 0", "обращений 2 из 20",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("в протоколе нет %q:\n%s", want, out)
		}
	}

	for _, leak := range []string{spokenText, testAPIKey, dir, "salute", "gemma"} {
		if strings.Contains(out, leak) {
			t.Errorf("в протокол попало %q:\n%s", leak, out)
		}
	}

	for name, want := range map[string]string{"a.txt": spokenText, "b.txt": ""} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || string(got) != want {
			t.Errorf("%s: %q, %v; ждали %q", name, got, err, want)
		}
	}
}

// TestSpeechRefusesBeforeCalls — потолок записей и негодный WAV отбивают прогон до первого обращения.
func TestSpeechRefusesBeforeCalls(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeSpeechKit(t, &calls)
	config := writeSpeechConfig(t, srv.URL)
	good := goodWAV(tone(testRate))

	cases := map[string]struct {
		clips    map[string][]byte
		maxFiles int
		task     string
		want     string
	}{
		"потолок записей": {map[string][]byte{"a.wav": good, "b.wav": good, "c.wav": good}, 2, "dictation",
			"записей 3, потолок 2"},
		"негодный WAV": {map[string][]byte{"a.wav": good, "bad.wav": []byte("mp3")}, 5, "dictation",
			"bad.wav: не RIFF/WAVE"},
		"нет записей":       {map[string][]byte{"a.txt": good}, 5, "dictation", "нет записей"},
		"задачи не выбрать": {map[string][]byte{"a.wav": good}, 5, "", "задач 2, нужна одна"},
		"чатовая задача":    {map[string][]byte{"a.wav": good}, 5, "ask", `"ask" не объявлена`},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			o := speechOpts(config, writeClips(t, tc.clips))
			o.maxFiles, o.task = tc.maxFiles, tc.task

			var stdout, stderr bytes.Buffer

			code := runSpeech(context.Background(), o, &stdout, &stderr, speechEnv)
			if code != exitUsage || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("код %d, stderr %q, ждали %q", code, stderr.String(), tc.want)
			}
		})
	}

	if n := calls.Load(); n != 0 {
		t.Fatalf("обращений к плечу %d, ждали ноль", n)
	}
}

func TestSpeechRequestCap(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeSpeechKit(t, &calls)
	dir := writeClips(t, map[string][]byte{"a.wav": goodWAV(tone(testRate)), "b.wav": goodWAV(tone(testRate))})
	o := speechOpts(writeSpeechConfig(t, srv.URL), dir)
	o.maxRequests = 1

	var stdout, stderr bytes.Buffer

	if code := runSpeech(context.Background(), o, &stdout, &stderr, speechEnv); code != exitIncomplete {
		t.Fatalf("код %d, ждали %d:\n%s", code, exitIncomplete, stdout.String())
	}

	if n := calls.Load(); n != 1 {
		t.Fatalf("обращений %d, ждали одно", n)
	}

	if out := stdout.String(); !strings.Contains(out, "- [skip] speech b.wav") {
		t.Fatalf("вторая запись не пропущена:\n%s", out)
	}
}

func TestSpeechProviderFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)

	dir := writeClips(t, map[string][]byte{"a.wav": goodWAV(tone(testRate))})

	var stdout, stderr bytes.Buffer

	code := runSpeech(context.Background(), speechOpts(writeSpeechConfig(t, srv.URL), dir), &stdout, &stderr, speechEnv)
	if code != exitViolation || !strings.Contains(stdout.String(), "HTTP 401") {
		t.Fatalf("код %d:\n%s", code, stdout.String())
	}

	if _, err := os.Stat(filepath.Join(dir, "a.txt")); !os.IsNotExist(err) {
		t.Fatalf("текст отказа записан: %v", err)
	}
}

func TestParseSpeechFlags(t *testing.T) {
	t.Parallel()

	o, err := parseSpeechFlags([]string{"-config", "c", "-dir", "d"}, io.Discard)
	if err != nil || o.maxFiles != defaultMaxFiles || o.task != "" {
		t.Fatalf("%+v, %v", o, err)
	}

	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"-dir", "d"}, "-config"},
		{[]string{"-config", "c"}, "-dir"},
		{[]string{"-config", "c", "-dir", "d", "-max-files", "0"}, "-max-files"},
	} {
		if _, flagErr := parseSpeechFlags(
			tc.args,
			io.Discard,
		); flagErr == nil ||
			!strings.Contains(flagErr.Error(), tc.want) {
			t.Errorf("%v: %v, ждали %q", tc.args, flagErr, tc.want)
		}
	}
}

// TestChatIgnoresSpeech — чатовые проверки не собирают плечи речи и не читают их ключи.
func TestChatIgnoresSpeech(t *testing.T) {
	t.Parallel()

	var calls int

	srv := fakeOllama(t, &calls)
	config := writeFile(t, `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "`+srv.URL+`"},
    "stt": {"kind": "speechkit", "endpoint": "https://stt.invalid", "folder": "f", "auth": {"api_key": "${UNSET}"}}
  },
  "tasks": {"ask": {"provider": "local", "model": "gemma", "output": {"mode": "json"}}},
  "speech": {"dictation": {"provider": "stt", "model": "general"}}
}`)

	var stdout, stderr bytes.Buffer

	code := dispatch(context.Background(), []string{"-config", config, "-checks", checkModel}, &stdout, &stderr, noEnv)
	if code != exitOK || strings.Contains(stdout.String(), "stt") {
		t.Fatalf("код %d, stderr %s\n%s", code, stderr.String(), stdout.String())
	}
}
