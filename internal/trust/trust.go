// Package trust — HTTP-клиент с корнями TLS целиком, без системных, на клоне транспорта основы.
package trust

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net/http"
)

// Client — основа с корнями ca; проверка TLS включается, даже если основа её выключила, прочий
// TLSClientConfig основы сохраняется. Пустой ca — основа как есть, nil-основа — [http.DefaultClient].
func Client(base *http.Client, ca *x509.CertPool) (*http.Client, error) {
	if base == nil {
		base = http.DefaultClient
	}

	if ca == nil {
		return base, nil
	}

	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}

	tr, ok := rt.(*http.Transport)
	if !ok {
		return nil, errors.New("CA задан, а транспорт не *http.Transport")
	}

	tr = tr.Clone()
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}

	tr.TLSClientConfig.RootCAs = ca
	tr.TLSClientConfig.InsecureSkipVerify = false

	client := *base
	client.Transport = tr

	return &client, nil
}
