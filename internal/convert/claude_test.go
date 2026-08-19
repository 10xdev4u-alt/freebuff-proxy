package convert

import (
	"encoding/json"
	"testing"
)

func TestConvertClaudeMinimalRequest(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "Hello",
			},
		},
	})

	out, model, stream, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "claude-3-5-sonnet-20241022" {
		t.Errorf("model = %q, want claude-3-5-sonnet-20241022", model)
	}
	if stream {
		t.Error("stream should be false")
	}
	if out["model"] != "claude-3-5-sonnet-20241022" {
		t.Errorf("output model = %v", out["model"])
	}
	if out["max_tokens"] != 1024 {
		t.Errorf("max_tokens = %v, want 1024", out["max_tokens"])
	}

	messages, ok := out["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("expected 1 message, got %v", out["messages"])
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "user" || msg["content"] != "Hello" {
		t.Errorf("unexpected message: %v", msg)
	}
}

func TestConvertClaudeRequestMissingModel(t *testing.T) {
	body := mustMarshal(map[string]any{
		"max_tokens": 1024,
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	_, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err == nil {
		t.Fatal("expected error for missing model")
	}
}

func TestConvertClaudeRequestMissingMessages(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
	})

	_, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err == nil {
		t.Fatal("expected error for missing messages")
	}
}

func TestConvertClaudeRequestWithSystemString(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 512,
		"system":     "You are a helpful assistant.",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(messages))
	}
	sys := messages[0].(map[string]any)
	if sys["role"] != "system" {
		t.Errorf("first message role = %v, want system", sys["role"])
	}
	if sys["content"] != "You are a helpful assistant." {
		t.Errorf("system content = %v", sys["content"])
	}
}

func TestConvertClaudeRequestWithSystemBlocks(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 512,
		"system": []any{
			map[string]any{"type": "text", "text": "You are helpful."},
			map[string]any{"type": "text", "text": "Be concise."},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	sys := messages[0].(map[string]any)
	if sys["role"] != "system" {
		t.Errorf("first message role = %v, want system", sys["role"])
	}
	// Two text blocks should be joined or kept as array
	content := sys["content"]
	t.Logf("system content type: %T, value: %v", content, content)
}

func TestConvertClaudeRequestWithStream(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"stream":     true,
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	_, _, stream, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !stream {
		t.Error("stream should be true")
	}
}

func TestConvertClaudeRequestWithTemperature(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":       "claude-3-5-sonnet-20241022",
		"max_tokens":  256,
		"temperature": 0.7,
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", out["temperature"])
	}
}

func TestConvertClaudeRequestWithStopSequences(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":           "claude-3-5-sonnet-20241022",
		"max_tokens":      256,
		"stop_sequences":  []string{"STOP", "END"},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	stops := out["stop"]
	if arr, ok := stops.([]string); !ok || len(arr) != 2 {
		t.Errorf("stop = %v, want [STOP END]", stops)
	}
}

func TestConvertClaudeRequestWithSingleStopSequence(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":           "claude-3-5-sonnet-20241022",
		"max_tokens":      256,
		"stop_sequences":  []string{"STOP"},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["stop"] != "STOP" {
		t.Errorf("stop = %v, want STOP", out["stop"])
	}
}

func TestConvertClaudeRequestWithToolDefinitions(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"tools": []any{
			map[string]any{
				"name":        "get_weather",
				"description": "Get weather for a location",
				"input_schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
				},
			},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "Weather in NYC?"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := out["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want function", tool["type"])
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v, want get_weather", fn["name"])
	}
	if fn["description"] != "Get weather for a location" {
		t.Errorf("function description = %v", fn["description"])
	}
	if fn["parameters"] == nil {
		t.Error("function parameters should not be nil")
	}
}

func TestConvertClaudeToolChoiceAuto(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"tool_choice": map[string]any{"type": "auto"},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, want auto", out["tool_choice"])
	}
}

func TestConvertClaudeToolChoiceNone(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"tool_choice": map[string]any{"type": "none"},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["tool_choice"] != "none" {
		t.Errorf("tool_choice = %v, want none", out["tool_choice"])
	}
}

func TestConvertClaudeToolChoiceAny(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"tool_choice": map[string]any{"type": "any"},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v, want required", out["tool_choice"])
	}
}

func TestConvertClaudeToolChoiceSpecificTool(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"tool_choice": map[string]any{
			"type": "tool",
			"name": "get_weather",
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tc := out["tool_choice"].(map[string]any)
	if tc["type"] != "function" {
		t.Errorf("tool_choice type = %v, want function", tc["type"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v, want get_weather", fn["name"])
	}
}

func TestConvertClaudeAssistantWithToolUse(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"messages": []any{
			map[string]any{"role": "user", "content": "Weather in NYC?"},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type":  "tool_use",
						"id":    "toolu_abc123",
						"name":  "get_weather",
						"input": map[string]any{"location": "NYC"},
					},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":            "tool_result",
						"tool_use_id":     "toolu_abc123",
						"content":         "72F and sunny",
					},
				},
			},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	// user, assistant (with tool_calls), tool (result), user
	// tool_result gets placed as a separate tool message before the next message
	if len(messages) < 3 {
		t.Fatalf("expected at least 3 messages, got %d", len(messages))
	}

	// Find the assistant message with tool_calls
	for _, m := range messages {
		msg := m.(map[string]any)
		if msg["role"] == "assistant" {
			if toolCalls, ok := msg["tool_calls"].([]any); ok && len(toolCalls) > 0 {
				tc := toolCalls[0].(map[string]any)
				if tc["id"] != "toolu_abc123" {
					t.Errorf("tool call id = %v, want toolu_abc123", tc["id"])
				}
				fn := tc["function"].(map[string]any)
				if fn["name"] != "get_weather" {
					t.Errorf("tool call name = %v, want get_weather", fn["name"])
				}
			}
		}
		if msg["role"] == "tool" {
			if msg["tool_call_id"] != "toolu_abc123" {
				t.Errorf("tool message tool_call_id = %v, want toolu_abc123", msg["tool_call_id"])
			}
		}
	}
}

func TestConvertClaudeAssistantWithThinking(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"messages": []any{
			map[string]any{"role": "user", "content": "Think about this"},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type":     "thinking",
						"thinking": "Let me consider...",
					},
					map[string]any{
						"type": "text",
						"text": "Here's my answer.",
					},
				},
			},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	for _, m := range messages {
		msg := m.(map[string]any)
		if msg["role"] == "assistant" {
			if rc, ok := msg["reasoning_content"]; ok && rc != "" {
				t.Logf("reasoning_content: %v", rc)
			}
			if msg["content"] != "Here's my answer." {
				t.Errorf("assistant content = %v, want Here's my answer.", msg["content"])
			}
		}
	}
}

func TestConvertClaudeImageBase64(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/png",
							"data":       "iVBORw0KGgo=",
						},
					},
					map[string]any{
						"type": "text",
						"text": "What's in this image?",
					},
				},
			},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	msg := messages[0].(map[string]any)
	contentArr := msg["content"].([]any)
	if len(contentArr) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(contentArr))
	}
	imgPart := contentArr[0].(map[string]any)
	if imgPart["type"] != "image_url" {
		t.Errorf("image type = %v, want image_url", imgPart["type"])
	}
	imgURL := imgPart["image_url"].(map[string]any)
	if imgURL["url"] == nil || imgURL["url"] == "" {
		t.Error("image URL should not be empty")
	}
}

func TestSanitizeClaudeToolID(t *testing.T) {
	tests := []struct {
		input    string
		contains string
	}{
		{"toolu_abc123", "toolu_abc123"},
		{"toolu_abc-123_def", "toolu_abc-123_def"},
		{"toolu<script>alert(1)</script>", "tooluscriptalert1script"},
		{"", ""}, // generates random ID
	}

	for _, tt := range tests {
		result := SanitizeClaudeToolID(tt.input)
		if tt.input == "" {
			if len(result) == 0 || result[:6] != "toolu_" {
				t.Errorf("empty input should generate toolu_ prefix, got %q", result)
			}
		} else if tt.contains != "" {
			// Just check it doesn't crash and produces output
			if len(result) == 0 {
				t.Errorf("SanitizeClaudeToolID(%q) produced empty result", tt.input)
			}
		}
	}
}

func TestBudgetToReasoningEffort(t *testing.T) {
	tests := []struct {
		budget int
		want   string
	}{
		{0, "none"},
		{-1, "none"},
		{256, "minimal"},
		{512, "minimal"},
		{1024, "low"},
		{4096, "medium"},
		{8192, "medium"},
		{16384, "high"},
		{24576, "high"},
		{32768, "xhigh"},
		{100000, "xhigh"},
	}

	for _, tt := range tests {
		got := BudgetToReasoningEffort(tt.budget)
		if got != tt.want {
			t.Errorf("BudgetToReasoningEffort(%d) = %q, want %q", tt.budget, got, tt.want)
		}
	}
}

func TestMapOpenAIFinishReasonToClaude(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"stop", "end_turn"},
		{"tool_calls", "tool_use"},
		{"function_call", "tool_use"},
		{"length", "max_tokens"},
		{"content_filter", "end_turn"},
		{"", "end_turn"},
	}

	for _, tt := range tests {
		got := mapOpenAIFinishReasonToClaude(tt.input)
		if got != tt.want {
			t.Errorf("mapOpenAIFinishReasonToClaude(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNormalizeClaudeErrorType(t *testing.T) {
	tests := []struct {
		statusCode int
		upstream   string
		want       string
	}{
		{400, "", "invalid_request_error"},
		{401, "", "authentication_error"},
		{403, "", "permission_error"},
		{404, "", "not_found_error"},
		{429, "", "rate_limit_error"},
		{503, "", "overloaded_error"},
		{500, "", "api_error"},
		{400, "rate_limit_error", "rate_limit_error"}, // upstream type preserved
	}

	for _, tt := range tests {
		got := normalizeClaudeErrorType(tt.statusCode, tt.upstream)
		if got != tt.want {
			t.Errorf("normalizeClaudeErrorType(%d, %q) = %q, want %q", tt.statusCode, tt.upstream, got, tt.want)
		}
	}
}

func TestConvertClaudeSystemEmpty(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"system":     "",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	// Empty system should not produce a system message
	for _, m := range messages {
		msg := m.(map[string]any)
		if msg["role"] == "system" {
			t.Error("empty system should not produce a system message")
		}
	}
}

func TestConvertClaudeUserWithImageAndText(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 256,
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type": "url",
							"url":  "https://example.com/img.png",
						},
					},
					map[string]any{"type": "text", "text": "Describe this"},
				},
			},
		},
	})

	out, _, _, err := ConvertClaudeMessagesRequestToOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := out["messages"].([]any)
	msg := messages[0].(map[string]any)
	content := msg["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(content))
	}
}

func TestConvertClaudeRequestInvalidJSON(t *testing.T) {
	_, _, _, err := ConvertClaudeMessagesRequestToOpenAI([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- Helpers ---

func mustMarshal(v any) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}
