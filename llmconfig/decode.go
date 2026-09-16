package llmconfig

import (
	"bytes"
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// strict разбирает ровно один документ JSON в dst и копит ошибки с путём: неизвестное поле, повтор
// ключа, чужой тип. encoding/json останавливается на первой и путь ключей карты не пишет.
func strict(data []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()

	var errs []error

	tree, err := readValue(dec, "", &errs)
	if err != nil {
		return fmt.Errorf("разбор JSON: %w", err)
	}

	if _, tailErr := dec.Token(); !errors.Is(tailErr, io.EOF) {
		return errors.New("разбор JSON: после документа есть продолжение")
	}

	checkValue("", tree, reflect.TypeOf(dst).Elem(), &errs)

	if len(errs) > 0 {
		return errors.Join(errs...)
	}

	return json.Unmarshal(data, dst)
}

// object — объект JSON: значения по ключам.
type object map[string]any

func readValue(dec *json.Decoder, path string, errs *[]error) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}

	switch tok {
	case json.Delim('{'):
		return readObject(dec, path, errs)
	case json.Delim('['):
		return readList(dec, path, errs)
	default:
		return tok, nil
	}
}

// readObject читает объект после `{`; повтор ключа — ошибка пути, значение берётся последнее.
func readObject(dec *json.Decoder, path string, errs *[]error) (object, error) {
	obj := object{}

	for dec.More() {
		keyTok, keyErr := dec.Token()
		if keyErr != nil {
			return nil, keyErr
		}

		key, _ := keyTok.(string)
		at := join(path, key)

		value, valueErr := readValue(dec, at, errs)
		if valueErr != nil {
			return nil, valueErr
		}

		if _, dup := obj[key]; dup {
			*errs = append(*errs, fmt.Errorf("%s: ключ повторён", at))
		}

		obj[key] = value
	}

	_, err := dec.Token()

	return obj, err
}

func readList(dec *json.Decoder, path string, errs *[]error) ([]any, error) {
	var list []any

	for i := 0; dec.More(); i++ {
		value, valueErr := readValue(dec, index(path, i), errs)
		if valueErr != nil {
			return nil, valueErr
		}

		list = append(list, value)
	}

	_, err := dec.Token()

	return list, err
}

// checkValue сверяет значение с типом приёмника по тегам json.
func checkValue(path string, v any, t reflect.Type, errs *[]error) {
	fail := func(want string) { *errs = append(*errs, fmt.Errorf("%s: ожидается %s", pathOf(path), want)) }

	switch kind := t.Kind(); {
	case reflect.PointerTo(t).Implements(reflect.TypeFor[encoding.TextUnmarshaler]()):
		if _, ok := v.(string); !ok {
			fail("строка")
		}
	case kind == reflect.Pointer:
		if v != nil {
			checkValue(path, v, t.Elem(), errs)
		}
	case kind == reflect.Struct:
		checkStruct(path, v, t, errs, fail)
	case kind == reflect.Map:
		obj, ok := v.(object)
		if !ok {
			fail("объект")

			return
		}

		for _, key := range slices.Sorted(maps.Keys(obj)) {
			checkValue(join(path, key), obj[key], t.Elem(), errs)
		}
	case kind == reflect.Slice:
		list, ok := v.([]any)
		if !ok {
			fail("список")

			return
		}

		for i, item := range list {
			checkValue(index(path, i), item, t.Elem(), errs)
		}
	default:
		if want := scalarMismatch(v, t); want != "" {
			fail(want)
		}
	}
}

func checkStruct(path string, v any, t reflect.Type, errs *[]error, fail func(string)) {
	obj, ok := v.(object)
	if !ok {
		fail("объект")

		return
	}

	fields := jsonFields(t)

	for _, key := range slices.Sorted(maps.Keys(obj)) {
		field, known := fields[key]
		if !known {
			*errs = append(*errs, fmt.Errorf("%s: неизвестное поле", join(path, key)))

			continue
		}

		checkValue(join(path, key), obj[key], field.Type, errs)
	}
}

// scalarMismatch — что ожидалось вместо v; пустое — значение подходит.
func scalarMismatch(v any, t reflect.Type) string {
	kind := t.Kind()
	n, isNumber := v.(json.Number)
	_, isString := v.(string)
	_, isBool := v.(bool)
	isInt := kind == reflect.Int || kind == reflect.Int64

	switch {
	case kind == reflect.String && !isString:
		return "строка"
	case kind == reflect.Bool && !isBool:
		return "true или false"
	case isInt && (!isNumber || !fitsInt(n, t.Bits())):
		return "целое число"
	case kind == reflect.Float64 && (!isNumber || !fitsFloat(n)):
		return "число"
	case kind != reflect.String && kind != reflect.Bool && !isInt && kind != reflect.Float64:
		return "значение поддержанного типа"
	default:
		return ""
	}
}

func fitsInt(n json.Number, bits int) bool {
	_, err := strconv.ParseInt(n.String(), 10, bits)

	return err == nil
}

func fitsFloat(n json.Number) bool {
	_, err := strconv.ParseFloat(n.String(), 64)

	return err == nil
}

// jsonFields — поля структуры по имени тега json; поле без тега не разбирается вовсе.
func jsonFields(t reflect.Type) map[string]reflect.StructField {
	fields := make(map[string]reflect.StructField, t.NumField())

	for i := range t.NumField() {
		f := t.Field(i)

		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if f.IsExported() && name != "" && name != "-" {
			fields[name] = f
		}
	}

	return fields
}

func join(path, key string) string {
	if path == "" {
		return key
	}

	return path + "." + key
}

func index(path string, i int) string { return fmt.Sprintf("%s[%d]", path, i) }

func pathOf(path string) string {
	if path == "" {
		return "документ"
	}

	return path
}
