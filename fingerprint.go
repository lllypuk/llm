package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// Fingerprint — короткий отпечаток набора строк, из которых складывается идентичность
// вызова: плечо, модель, ревизия, хеш промпта, схема, опции, подготовка входа.
// Считается до вызова — по нему дедупятся прогоны; порядок частей значим.
func Fingerprint(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))

	return hex.EncodeToString(sum[:8])
}
