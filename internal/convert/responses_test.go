package convert

import (
	"encoding/json"
	"testing"
)

func TestConvertResponsesMinimal(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
	})

	out, model, stream, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "gpt-4o" {
		t.Errorf("model = %q, want gpt-4o", model)
	}
	if stream {
		t.Error("stream should be false")
	}
	if out["model"] != "gpt-4o" {
		t.Errorf("output model = %v", out["model"])
	}

	messages := out["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("role = %v, want user", msg["role"])
	}
	if msg["content"] != "Hello" {
		t.Errorf("content = %v, want Hello", msg["content"])
	}
}

func TestConvertResponsesWithInstructions(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":        "gpt-4o",
		"input":        "What's the weather?",
		"instructions": "Be concise and helpful.",
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
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
	if sys["content"] != "Be concise and helpful." {
		t.Errorf("system content = %v", sys["content"])
	}
}

func TestConvertResponsesMissingModel(t *testing.T) {
	body := mustMarshal(map[string]any{
		"input": "Hello",
	})

	out, model, _, err := ConvertResponsesToChatParams(body)
	// Should return nil without error (graceful handling)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != nil {
		t.Error("expected nil output for missing model")
	}
	if model != "" {
		t.Errorf("model = %q, want empty", model)
	}
}

func TestConvertResponsesInvalidJSON(t *testing.T) {
	out, _, _, err := ConvertResponsesToChatParams([]byte("not json"))
	// Should return nil without error (graceful handling)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != nil {
		t.Error("expected nil output for invalid JSON")
	}
}

func TestConvertResponsesWithMaxOutputTokens(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":              "gpt-4o",
		"input":              "Hello",
		"max_output_tokens":  2048,
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["max_completion_tokens"] != 2048 {
		t.Errorf("max_completion_tokens = %v, want 2048", out["max_completion_tokens"])
	}
}

func TestConvertResponsesWithReasoning(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "o3",
		"input": "Think about this",
		"reasoning": map[string]any{
			"effort": "high",
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", out["reasoning_effort"])
	}
}

func TestConvertResponsesWithTextFormat(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Give me JSON",
		"text": map[string]any{
			"format": map[string]any{
				"type": "json_schema",
				"json_schema": map[string]any{
					"name":   "weather",
					"schema": map[string]any{"type": "object"},
				},
			},
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rf := out["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Errorf("response_format type = %v, want json_schema", rf["type"])
	}
	if rf["json_schema"] == nil {
		t.Error("json_schema should not be nil")
	}
}

func TestConvertResponsesTextFormatDefaultSkipped(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
		"text": map[string]any{
			"format": map[string]any{
				"type": "text",
			},
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := out["response_format"]; ok {
		t.Error("text format type should be skipped (it's the default)")
	}
}

func TestConvertResponsesPassthroughFields(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model":              "gpt-4o",
		"input":              "Hello",
		"temperature":        0.5,
		"top_p":              0.9,
		"seed":               42,
		"parallel_tool_calls": false,
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["temperature"] != 0.5 {
		t.Errorf("temperature = %v", out["temperature"])
	}
	if out["top_p"] != 0.9 {
		t.Errorf("top_p = %v", out["top_p"])
	}
}

func TestConvertResponsesStream(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
		"stream": true,
	})

	_, _, stream, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !stream {
		t.Error("stream should be true")
	}
}

func TestResponsesInputToMessages_String(t *testing.T) {
	messages := ResponsesInputToMessages("Hello world", "")
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "user" || msg["content"] != "Hello world" {
		t.Errorf("unexpected message: %v", msg)
	}
}

func TestResponsesInputToMessages_StringWithInstructions(t *testing.T) {
	messages := ResponsesInputToMessages("Hi", "Be helpful")
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	sys := messages[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "Be helpful" {
		t.Errorf("unexpected system: %v", sys)
	}
}

func TestResponsesInputToMessages_Array(t *testing.T) {
	input := []any{
		map[string]any{"type": "message", "role": "user", "content": "Hi"},
		map[string]any{"type": "message", "role": "assistant", "content": "Hello!"},
		map[string]any{"type": "message", "role": "user", "content": "How are you?"},
	}
	messages := ResponsesInputToMessages(input, "")
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}
}

func TestResponsesInputToMessages_FunctionCallOutput(t *testing.T) {
	input := []any{
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_abc123",
			"output":  `{"temp":72}`,
		},
	}
	messages := ResponsesInputToMessages(input, "")
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "tool" {
		t.Errorf("role = %v, want tool", msg["role"])
	}
	if msg["tool_call_id"] != "call_abc123" {
		t.Errorf("tool_call_id = %v, want call_abc123", msg["tool_call_id"])
	}
}

func TestResponsesInputToMessages_FunctionCallOutput_NonString(t *testing.T) {
	input := []any{
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_abc",
			"output":  map[string]any{"result": 42},
		},
	}
	messages := ResponsesInputToMessages(input, "")
	msg := messages[0].(map[string]any)
	content := msg["content"].(string)
	if content == "" {
		t.Error("non-string output should be JSON-marshaled")
	}
}

func TestResponsesInputToMessages_SkipsFunctionCall(t *testing.T) {
	input := []any{
		map[string]any{"type": "function_call", "name": "foo"},
		map[string]any{"type": "message", "role": "user", "content": "Hi"},
	}
	messages := ResponsesInputToMessages(input, "")
	if len(messages) != 1 {
		t.Fatalf("expected 1 message (function_call skipped), got %d", len(messages))
	}
}

func TestResponsesInputToMessages_SkipsReasoning(t *testing.T) {
	input := []any{
		map[string]any{"type": "reasoning"},
		map[string]any{"type": "message", "role": "user", "content": "Hi"},
	}
	messages := ResponsesInputToMessages(input, "")
	if len(messages) != 1 {
		t.Fatalf("expected 1 message (reasoning skipped), got %d", len(messages))
	}
}

func TestResponsesInputToMessages_InputTextParts(t *testing.T) {
	input := []any{
		map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_text", "text": "Hello"},
			},
		},
	}
	messages := ResponsesInputToMessages(input, "")
	msg := messages[0].(map[string]any)
	content := msg["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(content))
	}
	part := content[0].(map[string]any)
	if part["type"] != "text" {
		t.Errorf("part type = %v, want text", part["type"])
	}
}

func TestResponsesInputToMessages_Nil(t *testing.T) {
	messages := ResponsesInputToMessages(nil, "")
	if len(messages) != 1 {
		t.Fatalf("expected 1 message for nil input, got %d", len(messages))
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("role = %v, want user", msg["role"])
	}
}

func TestConvertResponsesWithTools(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "What's the weather?",
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "get_weather",
				"description": "Get weather",
				"parameters": map[string]any{
					"type": "object",
				},
			},
			map[string]any{
				"type": "web_search",
			},
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := out["tools"].([]any)
	// Should only include function tools, not web_search
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool (function only), got %d", len(tools))
	}
	tool := tools[0].(map[string]any)
	fn := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("tool name = %v, want get_weather", fn["name"])
	}
}

func TestConvertResponsesToolChoiceFunction(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
		"tool_choice": map[string]any{
			"type": "function",
			"name": "get_weather",
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
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

func TestConvertResponsesToolChoiceNonFunction(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
		"tool_choice": map[string]any{
			"type": "web_search",
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, want auto for non-function type", out["tool_choice"])
	}
}

func TestCoalesce(t *testing.T) {
	if coalesce(nil, nil, nil) != nil {
		t.Error("all nil should return nil")
	}
	if coalesce(nil, "hello", nil) != "hello" {
		t.Error("should return first non-nil")
	}
	if coalesce(42) != 42 {
		t.Error("single value should return it")
	}
}

func TestConvertResponsesEmptyToolsFiltered(t *testing.T) {
	body := mustMarshal(map[string]any{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": []any{
			map[string]any{"type": "web_search"},
		},
	})

	out, _, _, err := ConvertResponsesToChatParams(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := out["tools"]; ok {
		t.Error("all non-function tools should be filtered out, no tools key expected")
	}
}

// Suppress unused import
var _ = json.Marshal
