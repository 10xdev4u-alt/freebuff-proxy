package server_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/10xdev4u-alt/freebuff-proxy/internal/testutil"
)

// --- /v1/messages (Claude) integration tests ---

func TestClaudeMessagesStream(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-c1", 100, `"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-c1", 100, `"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":5}}]`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"max_tokens": 256,
		"stream": true,
		"messages": [{"role": "user", "content": "Hello"}]
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte(body), map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	bodyStr := string(data)
	if !strings.Contains(bodyStr, "event: message_start") {
		t.Error("missing message_start event")
	}
	if !strings.Contains(bodyStr, "event: content_block_delta") {
		t.Error("missing content_block_delta event")
	}
	if !strings.Contains(bodyStr, "event: message_stop") {
		t.Error("missing message_stop event")
	}
}

func TestClaudeMessagesNonStream(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-c2", 101, `"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello!"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-c2", 101, `"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-c2", 101, `"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"max_tokens": 256,
		"stream": false,
		"messages": [{"role": "user", "content": "Hello"}]
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte(body), map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var claude map[string]any
	if err := json.Unmarshal(data, &claude); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if claude["type"] != "message" {
		t.Errorf("type = %v, want message", claude["type"])
	}
	if claude["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", claude["role"])
	}
}

func TestClaudeMessagesMissingModel(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := `{"max_tokens": 256, "messages": [{"role": "user", "content": "Hi"}]}`

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte(body), map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 400 {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}

	var errResp map[string]any
	if err := json.Unmarshal(data, &errResp); err != nil {
		t.Fatalf("failed to unmarshal error: %v", err)
	}
	if errResp["type"] != "error" {
		t.Errorf("type = %v, want error", errResp["type"])
	}
}

func TestClaudeMessagesUnauthorized(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}`

	resp, _ := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte(body), map[string]string{
		"Content-Type":      "application/json",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 401 {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestClaudeMessagesWithSystemPrompt(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-c3", 102, `"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello!"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-c3", 102, `"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-c3", 102, `"usage":{"prompt_tokens":10,"completion_tokens":5}`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"max_tokens": 256,
		"stream": false,
		"system": "You are a pirate.",
		"messages": [{"role": "user", "content": "Hello"}]
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte(body), map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}
}

func TestClaudeMessagesInvalidJSON(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/messages", []byte("not json"), map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 400 {
		t.Errorf("status = %d, want 400. body: %s", resp.StatusCode, string(data))
	}

	var errResp map[string]any
	json.Unmarshal(data, &errResp)
	if errResp["type"] != "error" {
		t.Errorf("type = %v, want error", errResp["type"])
	}
}

// --- /v1/responses integration tests ---

func TestResponsesStream(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-r1", 100, `"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-r1", 100, `"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":5}}]`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"input": "Hello",
		"stream": true
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte(body), map[string]string{
		"Content-Type": "application/json",
		"x-api-key":    "test-key",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
}

func TestResponsesNonStream(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-r2", 101, `"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello!"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-r2", 101, `"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-r2", 101, `"usage":{"prompt_tokens":10,"completion_tokens":5}`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"input": "Hello",
		"stream": false
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte(body), map[string]string{
		"Content-Type": "application/json",
		"x-api-key":    "test-key",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}
}

func TestResponsesWithInstructions(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()
	mock.ChatBody = testutil.SSEEvent(chunk("chatcmpl-r3", 102, `"choices":[{"index":0,"delta":{"role":"assistant","content":"OK"},"finish_reason":null}]`)) +
		testutil.SSEEvent(chunk("chatcmpl-r3", 102, `"choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":2}}]`))

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := fmt.Sprintf(`{
		"model": "%s",
		"input": "Do something",
		"instructions": "Be concise.",
		"stream": true
	}`, modelA)

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte(body), map[string]string{
		"Content-Type": "application/json",
		"x-api-key":    "test-key",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200. body: %s", resp.StatusCode, string(data))
	}
}

func TestResponsesMissingModel(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := `{"input": "Hello"}`

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte(body), map[string]string{
		"Content-Type": "application/json",
		"x-api-key":    "test-key",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 400 {
		t.Errorf("status = %d, want 400. body: %s", resp.StatusCode, string(data))
	}
}

func TestResponsesUnauthorized(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	body := `{"model":"gpt-4o","input":"Hello"}`

	resp, _ := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte(body), map[string]string{
		"Content-Type": "application/json",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 401 {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestResponsesInvalidJSON(t *testing.T) {
	mock := testutil.NewMock()
	defer mock.Close()

	ts, _ := newTestServer(t, []string{"test-key"}, mock)
	defer ts.Close()

	resp, data := doJSON(t, http.MethodPost, ts.URL+"/v1/responses", []byte("not json"), map[string]string{
		"Content-Type": "application/json",
		"x-api-key":    "test-key",
	})
	defer resp.Body.Close()

	if resp.StatusCode != 400 {
		t.Errorf("status = %d, want 400. body: %s", resp.StatusCode, string(data))
	}
}
