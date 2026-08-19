package convert

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestConvertOpenAINonStreamResponseToClaude_TextOnly(t *testing.T) {
	resp := map[string]any{
		"id":    "chatcmpl-abc123",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello, world!",
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	}

	body := mustMarshal(resp)
	result, err := convertOpenAINonStreamResponseToClaude(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var claude map[string]any
	if err := json.Unmarshal(result, &claude); err != nil {
		t.Fatalf("failed to unmarshal claude response: %v", err)
	}

	if claude["type"] != "message" {
		t.Errorf("type = %v, want message", claude["type"])
	}
	if claude["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", claude["role"])
	}
	if claude["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", claude["stop_reason"])
	}

	content := claude["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("content type = %v, want text", block["type"])
	}
	if block["text"] != "Hello, world!" {
		t.Errorf("content text = %v, want Hello, world!", block["text"])
	}
}

func TestConvertOpenAINonStreamResponseToClaude_ToolCalls(t *testing.T) {
	resp := map[string]any{
		"id":    "chatcmpl-abc123",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []any{
						map[string]any{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]any{
								"name":      "get_weather",
								"arguments": `{"location":"NYC"}`,
							},
						},
					},
				},
			},
		},
	}

	body := mustMarshal(resp)
	result, err := convertOpenAINonStreamResponseToClaude(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var claude map[string]any
	if err := json.Unmarshal(result, &claude); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if claude["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", claude["stop_reason"])
	}

	content := claude["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("content type = %v, want tool_use", block["type"])
	}
	if block["name"] != "get_weather" {
		t.Errorf("tool name = %v, want get_weather", block["name"])
	}
}

func TestConvertOpenAINonStreamResponseToClaude_Reasoning(t *testing.T) {
	resp := map[string]any{
		"id":    "chatcmpl-abc123",
		"model": "o3",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":             "assistant",
					"content":          "The answer is 42.",
					"reasoning_content": "Let me think step by step...",
				},
			},
		},
	}

	body := mustMarshal(resp)
	result, err := convertOpenAINonStreamResponseToClaude(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var claude map[string]any
	if err := json.Unmarshal(result, &claude); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	content := claude["content"].([]any)
	// Should have thinking + text blocks
	if len(content) < 2 {
		t.Fatalf("expected at least 2 content blocks (thinking + text), got %d", len(content))
	}
}

func TestWriteClaudeSSEEvents(t *testing.T) {
	w := httptest.NewRecorder()
	events := []ClaudeSSEEvent{
		{Name: "message_start", Payload: []byte(`{"type":"message_start"}`)},
		{Name: "content_block_start", Payload: []byte(`{"type":"content_block_start","index":0}`)},
	}

	err := WriteClaudeSSEEvents(w, events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "event: message_start") {
		t.Error("missing message_start event")
	}
	if !strings.Contains(body, "event: content_block_start") {
		t.Error("missing content_block_start event")
	}
	if !strings.Contains(body, `{"type":"message_start"}`) {
		t.Error("missing message_start payload")
	}
}

func TestWriteClaudeError(t *testing.T) {
	w := httptest.NewRecorder()
	WriteClaudeError(w, 400, "bad request", "invalid_request_error")

	body := w.Body.String()
	var resp map[string]any
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("failed to unmarshal error: %v", err)
	}

	if resp["type"] != "error" {
		t.Errorf("type = %v, want error", resp["type"])
	}
	errObj := resp["error"].(map[string]any)
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("error type = %v, want invalid_request_error", errObj["type"])
	}
	if errObj["message"] != "bad request" {
		t.Errorf("error message = %v, want bad request", errObj["message"])
	}
}

func TestWriteClaudePassthroughError_JSON(t *testing.T) {
	w := httptest.NewRecorder()
	upstream := []byte(`{"error":{"message":"rate limited","type":"rate_limit_error"}}`)
	WriteClaudePassthroughError(w, 429, upstream)

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	errObj := resp["error"].(map[string]any)
	if errObj["type"] != "rate_limit_error" {
		t.Errorf("error type = %v, want rate_limit_error", errObj["type"])
	}
}

func TestWriteClaudePassthroughError_PlainText(t *testing.T) {
	w := httptest.NewRecorder()
	WriteClaudePassthroughError(w, 500, []byte("internal server error"))

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	errObj := resp["error"].(map[string]any)
	if errObj["type"] != "api_error" {
		t.Errorf("error type = %v, want api_error", errObj["type"])
	}
}

func TestStreamConversion_BasicText(t *testing.T) {
	state := NewClaudeStreamState("gpt-4o")

	// First chunk with content
	chunk1 := map[string]any{
		"id":    "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"role":    "assistant",
					"content": "Hello",
				},
			},
		},
	}
	payload1 := mustMarshal(chunk1)
	events1, err := ConvertOpenAIStreamPayloadToClaudeEvents(payload1, state)
	if err != nil {
		t.Fatalf("chunk1 error: %v", err)
	}

	// Should have: message_start, content_block_start (text), content_block_delta (text)
	if len(events1) < 3 {
		t.Fatalf("expected at least 3 events from chunk1, got %d", len(events1))
	}
	if events1[0].Name != "message_start" {
		t.Errorf("first event = %v, want message_start", events1[0].Name)
	}

	// Second chunk with more content and usage
	chunk2 := map[string]any{
		"id":    "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "stop",
				"delta": map[string]any{
					"content": " world!",
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 5,
		},
	}
	payload2 := mustMarshal(chunk2)
	events2, err := ConvertOpenAIStreamPayloadToClaudeEvents(payload2, state)
	if err != nil {
		t.Fatalf("chunk2 error: %v", err)
	}

	// Should have text delta + content_block_stop (from finish_reason) + message_delta + message_stop (from usage)
	foundStop := false
	foundDelta := false
	for _, e := range events2 {
		if e.Name == "message_stop" {
			foundStop = true
		}
		if e.Name == "message_delta" {
			foundDelta = true
		}
	}
	if !foundStop {
		t.Error("expected message_stop event in chunk2")
	}
	if !foundDelta {
		t.Error("expected message_delta event in chunk2")
	}
}

func TestStreamConversion_ToolCalls(t *testing.T) {
	state := NewClaudeStreamState("gpt-4o")

	// Tool call start
	chunk1 := map[string]any{
		"id":    "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"role": "assistant",
					"tool_calls": []any{
						map[string]any{
							"index": 0,
							"id":    "call_abc123",
							"type":  "function",
							"function": map[string]any{
								"name":      "get_weather",
								"arguments": "",
							},
						},
					},
				},
			},
		},
	}
	events1, err := ConvertOpenAIStreamPayloadToClaudeEvents(mustMarshal(chunk1), state)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Should have content_block_start for tool_use
	foundToolStart := false
	for _, e := range events1 {
		if e.Name == "content_block_start" {
			var payload map[string]any
			json.Unmarshal(e.Payload, &payload)
			cb := payload["content_block"].(map[string]any)
			if cb["type"] == "tool_use" {
				foundToolStart = true
			}
		}
	}
	if !foundToolStart {
		t.Error("expected tool_use content_block_start")
	}

	// Tool call arguments
	chunk2 := map[string]any{
		"id":    "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []any{
						map[string]any{
							"index": 0,
							"function": map[string]any{
								"arguments": `{"location":"NYC"}`,
							},
						},
					},
				},
			},
		},
	}
	_, err = ConvertOpenAIStreamPayloadToClaudeEvents(mustMarshal(chunk2), state)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Finish
	chunk3 := map[string]any{
		"id":    "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "tool_calls",
			},
		},
	}
	events3, err := ConvertOpenAIStreamPayloadToClaudeEvents(mustMarshal(chunk3), state)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Should have input_json_delta + content_block_stop for tool
	foundDelta := false
	for _, e := range events3 {
		if e.Name == "content_block_delta" {
			var payload map[string]any
			json.Unmarshal(e.Payload, &payload)
			if payload["delta"] != nil {
				delta := payload["delta"].(map[string]any)
				if delta["type"] == "input_json_delta" {
					foundDelta = true
				}
			}
		}
	}
	if !foundDelta {
		t.Error("expected input_json_delta for tool arguments")
	}
}

func TestStreamConversion_DoneEvent(t *testing.T) {
	state := NewClaudeStreamState("gpt-4o")
	state.MessageStarted = true
	state.TextContentStarted = true
	state.TextContentBlockIdx = 0

	events, err := FinalizeClaudeStream(state)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Should have content_block_stop + message_delta + message_stop
	names := make([]string, 0)
	for _, e := range events {
		names = append(names, e.Name)
	}

	hasStop := false
	hasDelta := false
	hasMsgStop := false
	for _, name := range names {
		if name == "content_block_stop" {
			hasStop = true
		}
		if name == "message_delta" {
			hasDelta = true
		}
		if name == "message_stop" {
			hasMsgStop = true
		}
	}
	if !hasStop {
		t.Error("expected content_block_stop")
	}
	if !hasDelta {
		t.Error("expected message_delta")
	}
	if !hasMsgStop {
		t.Error("expected message_stop")
	}
}

func TestWriteClaudeStreamingResponse_FromReader(t *testing.T) {
	// Simulate an OpenAI SSE stream
	var buf bytes.Buffer
	chunks := []string{
		`data: {"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}` + "\n\n",
		`data: {"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":5}}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	for _, c := range chunks {
		buf.WriteString(c)
	}

	w := httptest.NewRecorder()
	w.Header().Set("Content-Type", "text/event-stream")

	err := WriteClaudeStreamingResponseFromReader(w, &buf, "claude-3-5-sonnet-20241022")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	// Verify Claude SSE format
	if !strings.Contains(body, "event: message_start") {
		t.Error("missing message_start")
	}
	if !strings.Contains(body, "event: content_block_start") {
		t.Error("missing content_block_start")
	}
	if !strings.Contains(body, "event: content_block_delta") {
		t.Error("missing content_block_delta")
	}
	if !strings.Contains(body, "event: message_stop") {
		t.Error("missing message_stop")
	}
}

func TestCollectReasoningTexts(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"string", `"Let me think"`, 1},
		{"array of strings", `["step 1","step 2"]`, 2},
		{"null", `null`, 0},
		{"empty", `""`, 0},
		{"objects with text", `[{"text":"think"},{"text":"more"}]`, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := json.RawMessage(tt.input)
			got := collectReasoningTexts(raw)
			if len(got) != tt.want {
				t.Errorf("got %d texts, want %d: %v", len(got), tt.want, got)
			}
		})
	}
}

func TestExtractOpenAIUsage(t *testing.T) {
	tests := []struct {
		name         string
		usage        *openAIUsage
		wantInput    int64
		wantOutput   int64
		wantCached   int64
	}{
		{
			"nil usage", nil, 0, 0, 0,
		},
		{
			"no cached",
			&openAIUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
			100, 50, 0,
		},
		{
			"with cached",
			&openAIUsage{
				PromptTokens: 100, CompletionTokens: 50,
				PromptTokensDetails: &openAIPromptTokensDetail{CachedTokens: 30},
			},
			70, 50, 30, // input = 100 - 30 = 70
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, output, cached := extractOpenAIUsage(tt.usage)
			if input != tt.wantInput || output != tt.wantOutput || cached != tt.wantCached {
				t.Errorf("got (%d, %d, %d), want (%d, %d, %d)",
					input, output, cached, tt.wantInput, tt.wantOutput, tt.wantCached)
			}
		})
	}
}

// --- Test helper for non-stream write via HTTP ---

func TestWriteClaudeNonStreamResponse(t *testing.T) {
	resp := map[string]any{
		"id":    "chatcmpl-abc123",
		"model": "gpt-4o",
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": "Test response",
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     5,
			"completion_tokens": 3,
			"total_tokens":      8,
		},
	}

	respBody := mustMarshal(resp)
	httpResp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(respBody)),
	}

	w := httptest.NewRecorder()
	err := WriteClaudeNonStreamResponse(w, httpResp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if w.Code != 200 {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var claude map[string]any
	json.Unmarshal(w.Body.Bytes(), &claude)
	if claude["type"] != "message" {
		t.Errorf("type = %v, want message", claude["type"])
	}
}
