package llm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

// fingerprintVersion — версия кодирования: смена меняет все отпечатки, старые не пересчитываются.
const fingerprintVersion = 1

// Fingerprint — отпечаток набора строк, из которых складывается идентичность
// вызова: плечо, модель, ревизия, хеш промпта, схема, опции, подготовка входа.
// Части кодируются длиной, поэтому границы однозначны; порядок значим. 128 бит hex.
func Fingerprint(parts ...string) string {
	h := sha256.New()
	_ = binary.Write(h, binary.BigEndian, uint16(fingerprintVersion))

	for _, part := range parts {
		_ = binary.Write(h, binary.BigEndian, uint64(len(part)))
		_, _ = h.Write([]byte(part))
	}

	return hex.EncodeToString(h.Sum(nil)[:16])
}
