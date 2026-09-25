package llm_test

import (
	"errors"
	"testing"

	"github.com/lllypuk/llm"
)

// TestOCRRequestValidate — годный кадр проходит, каждый негодный отбивается RequestError до сети.
func TestOCRRequestValidate(t *testing.T) {
	t.Parallel()

	img := []byte{0xff, 0xd8, 0xff, 0xe0}

	cases := []struct {
		name    string
		req     llm.OCRRequest
		max     int
		wantErr bool
	}{
		{"годный JPEG", llm.OCRRequest{Model: "page", Image: img, MIME: llm.MIMEJPEG}, 10, false},
		{"годный PNG", llm.OCRRequest{Model: "page", Image: img, MIME: llm.MIMEPNG}, 0, false},
		{"ровно предел", llm.OCRRequest{Model: "page", Image: img, MIME: llm.MIMEJPEG}, len(img), false},
		{"без модели", llm.OCRRequest{Image: img, MIME: llm.MIMEJPEG}, 0, true},
		{"пустой кадр", llm.OCRRequest{Model: "page", MIME: llm.MIMEJPEG}, 0, true},
		{"чужой MIME", llm.OCRRequest{Model: "page", Image: img, MIME: "application/pdf"}, 0, true},
		{"без MIME", llm.OCRRequest{Model: "page", Image: img}, 0, true},
		{"больше предела", llm.OCRRequest{Model: "page", Image: img, MIME: llm.MIMEJPEG}, len(img) - 1, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := tc.req.Validate(tc.max)
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("Validate() = %v, ждали nil", err)
				}

				return
			}

			var reqErr *llm.RequestError
			if !errors.As(err, &reqErr) {
				t.Fatalf("Validate() = %v, ждали *RequestError", err)
			}
		})
	}
}
