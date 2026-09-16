//go:build !live

package llmcheck_test

import (
	"os"
	"os/exec"
	"testing"
)

// TestLiveBuild — команда собрана только тегом live, и без этого теста её поломку не заметил бы ни
// go test ./..., ни go vet: тесты под тегом гоняются вложенным go test без сети.
func TestLiveBuild(t *testing.T) {
	t.Parallel()

	cmd := exec.CommandContext(t.Context(), "go", "test", "-tags", "live", "-count", "1", ".")
	cmd.Env = append(os.Environ(), "GOFLAGS=")

	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go test -tags live: %v\n%s", err, out)
	}
}
