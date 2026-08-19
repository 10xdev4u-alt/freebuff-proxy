package convert

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ClaudeMessageParts holds the parsed components of a single Claude message
// after content block processing.
type ClaudeMessageParts struct {
	ContentParts   []any
	BeforeMessages []any
	AfterMessages  []any
	ToolCalls      []any
	Reasoning      string
}

// ConvertClaudeMessagesRequestToOpenAI transforms an Anthropic Messages API
// request body into an OpenAI chat-completions request body. Returns the
// converted payload, model name, stream flag, and any error.
func ConvertClaudeMessagesRequestToOpenAI(body []byte) (map[string]any, string, bool, error) {
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, "", false, fmt.Errorf("request body must be valid JSON")
	}

	modelName := strings.TrimSpace(toString(root["model"]))
	if modelName == "" {
		return nil, "", false, fmt.Errorf("model is required")
	}

	stream := toBool(root["stream"])
	out := map[string]any{
		"model":    modelName,
		"messages": []any{},
		"stream":   stream,
	}

	if maxTokens, ok := toInt(root["max_tokens"]); ok && maxTokens > 0 {
		out["max_tokens"] = maxTokens
	}

	if temperature, ok := toFloat(root["temperature"]); ok {
		out["temperature"] = temperature
	} else if topP, ok := toFloat(root["top_p"]); ok {
		out["top_p"] = topP
	}

	if stops := toStringSlice(root["stop_sequences"]); len(stops) > 0 {
		if len(stops) == 1 {
			out["stop"] = stops[0]
		} else {
			out["stop"] = stops
		}
	}

	if reasoningEffort, ok := mapClaudeThinkingToReasoningEffort(root); ok {
		out["reasoning_effort"] = reasoningEffort
	}

	messages := make([]any, 0, 8)
	if system := root["system"]; system != nil {
		if systemMessage := convertClaudeSystemToOpenAIMessage(system); systemMessage != nil {
			messages = append(messages, systemMessage)
		}
	}

	rawMessages := toSlice(root["messages"])
	if rawMessages == nil {
		return nil, "", false, fmt.Errorf("messages must be an array")
	}

	for _, rawMessage := range rawMessages {
		message := toMap(rawMessage)
		if message == nil {
			continue
		}

		role := strings.TrimSpace(toString(message["role"]))
		if role == "" {
			continue
		}

		parts := convertClaudeMessageContent(role, message["content"])
		for _, toolResult := range parts.BeforeMessages {
			messages = append(messages, toolResult)
		}

		switch role {
		case "assistant":
			if len(parts.ContentParts) == 0 && len(parts.ToolCalls) == 0 && parts.Reasoning == "" {
				continue
			}

			openAIMessage := map[string]any{"role": "assistant"}
			if len(parts.ContentParts) > 0 {
				openAIMessage["content"] = normalizeOpenAIContent(parts.ContentParts)
			} else {
				openAIMessage["content"] = ""
			}
			if parts.Reasoning != "" {
				openAIMessage["reasoning_content"] = parts.Reasoning
			}
			if len(parts.ToolCalls) > 0 {
				openAIMessage["tool_calls"] = parts.ToolCalls
			}
			messages = append(messages, openAIMessage)

		case "user":
			if len(parts.ContentParts) == 0 {
				continue
			}
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": normalizeOpenAIContent(parts.ContentParts),
			})
		}

		for _, toolResult := range parts.AfterMessages {
			messages = append(messages, toolResult)
		}
	}

	out["messages"] = messages

	builtinToolKinds := make(map[string]string)
	if rawTools := toSlice(root["tools"]); len(rawTools) > 0 {
		tools := make([]any, 0, len(rawTools))
		for _, rawTool := range rawTools {
			tool := toMap(rawTool)
			if tool == nil {
				continue
			}

			mappedTool, builtinKind := convertClaudeToolDefinitionToOpenAI(tool)
			if mappedTool == nil {
				continue
			}

			if builtinKind != "" {
				if toolName := strings.TrimSpace(toString(tool["name"])); toolName != "" {
					builtinToolKinds[toolName] = builtinKind
				}
			}

			tools = append(tools, mappedTool)
		}
		if len(tools) > 0 {
			out["tools"] = tools
		}
	}

	if toolChoice := toMap(root["tool_choice"]); toolChoice != nil {
		if mappedToolChoice, ok := convertClaudeToolChoiceToOpenAI(toolChoice, builtinToolKinds); ok {
			out["tool_choice"] = mappedToolChoice
		}
	}

	if userValue := strings.TrimSpace(toString(root["user"])); userValue != "" {
		out["user"] = userValue
	}

	return out, modelName, stream, nil
}

// convertClaudeToolDefinitionToOpenAI maps a Claude tool definition to OpenAI format.
func convertClaudeToolDefinitionToOpenAI(tool map[string]any) (map[string]any, string) {
	toolType := strings.ToLower(strings.TrimSpace(toString(tool["type"])))
	if mappedType, ok := mapClaudeBuiltinToolType(toolType); ok {
		mapped := cloneMap(tool)
		mapped["type"] = mappedType
		delete(mapped, "name")
		return mapped, mappedType
	}

	function := map[string]any{
		"name":        toString(tool["name"]),
		"description": toString(tool["description"]),
	}

	if inputSchema := tool["input_schema"]; inputSchema != nil {
		function["parameters"] = inputSchema
	}

	return map[string]any{
		"type":     "function",
		"function": function,
	}, ""
}

// convertClaudeToolChoiceToOpenAI maps Claude tool_choice to OpenAI format.
func convertClaudeToolChoiceToOpenAI(toolChoice map[string]any, builtinToolKinds map[string]string) (any, bool) {
	switch strings.ToLower(strings.TrimSpace(toString(toolChoice["type"]))) {
	case "none":
		return "none", true
	case "auto":
		return "auto", true
	case "any":
		return "required", true
	case "tool":
		toolName := strings.TrimSpace(toString(toolChoice["name"]))
		if toolName == "" {
			return nil, false
		}
		if builtinType := builtinToolKinds[toolName]; builtinType != "" {
			return map[string]any{"type": builtinType}, true
		}
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName,
			},
		}, true
	default:
		return nil, false
	}
}

// mapClaudeBuiltinToolType maps Claude tool types to OpenAI types.
func mapClaudeBuiltinToolType(toolType string) (string, bool) {
	switch toolType {
	case "web_search_20250305", "web_search":
		return "web_search", true
	default:
		return "", false
	}
}

// convertClaudeSystemToOpenAIMessage converts the Claude system field to an
// OpenAI system message.
func convertClaudeSystemToOpenAIMessage(system any) map[string]any {
	switch typed := system.(type) {
	case string:
		text := strings.TrimSpace(typed)
		if text == "" {
			return nil
		}
		return map[string]any{
			"role":    "system",
			"content": text,
		}
	case []any:
		contentParts := make([]any, 0, len(typed))
		for _, rawPart := range typed {
			part := toMap(rawPart)
			if part == nil {
				continue
			}
			if strings.EqualFold(toString(part["type"]), "text") {
				text := toString(part["text"])
				if strings.TrimSpace(text) == "" {
					continue
				}
				contentParts = append(contentParts, map[string]any{
					"type": "text",
					"text": text,
				})
			}
		}
		if len(contentParts) == 0 {
			return nil
		}
		return map[string]any{
			"role":    "system",
			"content": normalizeOpenAIContent(contentParts),
		}
	default:
		return nil
	}
}

// convertClaudeMessageContent processes a single Claude message's content
// blocks into OpenAI-compatible components.
func convertClaudeMessageContent(role string, content any) ClaudeMessageParts {
	result := ClaudeMessageParts{
		ContentParts:   make([]any, 0),
		BeforeMessages: make([]any, 0),
		AfterMessages:  make([]any, 0),
		ToolCalls:      make([]any, 0),
	}

	switch typed := content.(type) {
	case string:
		if strings.TrimSpace(typed) == "" {
			return result
		}
		result.ContentParts = append(result.ContentParts, map[string]any{"type": "text", "text": typed})
		return result
	case []any:
		reasoningParts := make([]string, 0)

		for _, rawPart := range typed {
			part := toMap(rawPart)
			if part == nil {
				continue
			}

			switch strings.ToLower(strings.TrimSpace(toString(part["type"]))) {
			case "text":
				text := toString(part["text"])
				if strings.TrimSpace(text) == "" {
					continue
				}
				result.ContentParts = append(result.ContentParts, map[string]any{
					"type": "text",
					"text": text,
				})

			case "image":
				if imagePart := convertClaudeImagePartToOpenAI(part); imagePart != nil {
					result.ContentParts = append(result.ContentParts, imagePart)
				}

			case "tool_use", "server_tool_use":
				if role != "assistant" {
					continue
				}
				toolCallID := SanitizeClaudeToolID(toString(part["id"]))
				result.ToolCalls = append(result.ToolCalls, map[string]any{
					"id":   toolCallID,
					"type": "function",
					"function": map[string]any{
						"name":      toString(part["name"]),
						"arguments": marshalJSONObject(part["input"]),
					},
				})

			case "tool_result":
				if toolMessage := buildOpenAIToolResultMessage(part["tool_use_id"], part["content"]); toolMessage != nil {
					result.BeforeMessages = append(result.BeforeMessages, toolMessage)
				}

			case "thinking":
				if role != "assistant" {
					continue
				}
				thinkingText := strings.TrimSpace(firstNonEmptyStr(part["thinking"], part["text"]))
				if thinkingText != "" {
					reasoningParts = append(reasoningParts, thinkingText)
				}

			default:
				partType := strings.ToLower(strings.TrimSpace(toString(part["type"])))
				switch {
				case strings.HasSuffix(partType, "_tool_use"):
					if role != "assistant" {
						continue
					}
					toolCallID := SanitizeClaudeToolID(firstNonEmptyStr(part["tool_use_id"], part["id"]))
					result.ToolCalls = append(result.ToolCalls, map[string]any{
						"id":   toolCallID,
						"type": "function",
						"function": map[string]any{
							"name":      firstNonEmptyStr(part["name"], part["tool_name"]),
							"arguments": marshalJSONObject(part["input"]),
						},
					})
				case strings.HasSuffix(partType, "_tool_result"):
					toolContent := any(cloneMap(part))
					if contentValue, ok := part["content"]; ok {
						toolContent = contentValue
					}
					if toolMessage := buildOpenAIToolResultMessage(firstNonEmptyStr(part["tool_use_id"], part["id"]), toolContent); toolMessage != nil {
						if role == "assistant" {
							result.AfterMessages = append(result.AfterMessages, toolMessage)
						} else {
							result.BeforeMessages = append(result.BeforeMessages, toolMessage)
						}
					}
				default:
					if fallbackPart := convertClaudeUnknownContentPartToOpenAI(part); fallbackPart != nil {
						result.ContentParts = append(result.ContentParts, fallbackPart)
					}
				}
			}
		}

		result.Reasoning = strings.Join(reasoningParts, "\n\n")
		return result
	default:
		return result
	}
}

// buildOpenAIToolResultMessage creates an OpenAI tool message from a Claude
// tool_use_id and content.
func buildOpenAIToolResultMessage(toolUseID any, content any) map[string]any {
	rawToolUseID := strings.TrimSpace(toString(toolUseID))
	if rawToolUseID == "" {
		return nil
	}
	sanitizedToolUseID := SanitizeClaudeToolID(rawToolUseID)

	return map[string]any{
		"role":         "tool",
		"tool_call_id": sanitizedToolUseID,
		"content":      convertClaudeToolResultContentToOpenAI(content),
	}
}

// convertClaudeUnknownContentPartToOpenAI JSON-encodes an unrecognized
// content part as a text block.
func convertClaudeUnknownContentPartToOpenAI(part map[string]any) map[string]any {
	encoded, err := json.Marshal(part)
	if err != nil || len(encoded) == 0 {
		return nil
	}

	return map[string]any{
		"type": "text",
		"text": string(encoded),
	}
}

// convertClaudeImagePartToOpenAI converts a Claude image content part to
// OpenAI image_url format.
func convertClaudeImagePartToOpenAI(part map[string]any) map[string]any {
	imageURL := ""

	if source := toMap(part["source"]); source != nil {
		switch strings.ToLower(strings.TrimSpace(toString(source["type"]))) {
		case "base64":
			data := toString(source["data"])
			if data != "" {
				mediaType := toString(source["media_type"])
				if mediaType == "" {
					mediaType = "application/octet-stream"
				}
				imageURL = "data:" + mediaType + ";base64," + data
			}
		case "url":
			imageURL = toString(source["url"])
		}
	}

	if imageURL == "" {
		imageURL = toString(part["url"])
	}
	if strings.TrimSpace(imageURL) == "" {
		return nil
	}

	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": imageURL,
		},
	}
}

// convertClaudeToolResultContentToOpenAI converts Claude tool_result content
// to OpenAI format (string, text array, or structured content).
func convertClaudeToolResultContentToOpenAI(content any) any {
	switch typed := content.(type) {
	case nil:
		return ""
	case string:
		return typed
	case []any:
		parts := make([]string, 0, len(typed))
		contentParts := make([]any, 0, len(typed))
		hasStructuredPart := false

		for _, rawItem := range typed {
			switch item := rawItem.(type) {
			case string:
				text := item
				parts = append(parts, text)
				contentParts = append(contentParts, map[string]any{"type": "text", "text": text})
			case map[string]any:
				switch strings.ToLower(strings.TrimSpace(toString(item["type"]))) {
				case "text":
					text := toString(item["text"])
					parts = append(parts, text)
					contentParts = append(contentParts, map[string]any{"type": "text", "text": text})
				case "image":
					if imagePart := convertClaudeImagePartToOpenAI(item); imagePart != nil {
						contentParts = append(contentParts, imagePart)
						hasStructuredPart = true
					}
				default:
					hasStructuredPart = true
					encoded, _ := json.Marshal(item)
					parts = append(parts, string(encoded))
				}
			default:
				encoded, _ := json.Marshal(item)
				parts = append(parts, string(encoded))
			}
		}

		if hasStructuredPart {
			if len(contentParts) > 0 {
				return normalizeOpenAIContent(contentParts)
			}
			encoded, _ := json.Marshal(typed)
			return string(encoded)
		}

		if len(contentParts) > 0 {
			return normalizeOpenAIContent(contentParts)
		}
		return strings.Join(parts, "\n\n")
	case map[string]any:
		switch strings.ToLower(strings.TrimSpace(toString(typed["type"]))) {
		case "text":
			return toString(typed["text"])
		case "image":
			if imagePart := convertClaudeImagePartToOpenAI(typed); imagePart != nil {
				return []any{imagePart}
			}
		}
		encoded, _ := json.Marshal(typed)
		return string(encoded)
	default:
		encoded, _ := json.Marshal(typed)
		return string(encoded)
	}
}

// mapClaudeThinkingToReasoningEffort extracts reasoning effort from Claude
// thinking configuration.
func mapClaudeThinkingToReasoningEffort(root map[string]any) (string, bool) {
	thinking := toMap(root["thinking"])
	if thinking == nil {
		return "", false
	}

	switch strings.ToLower(strings.TrimSpace(toString(thinking["type"]))) {
	case "disabled":
		return "none", true
	case "enabled":
		if budget, ok := toInt(thinking["budget_tokens"]); ok {
			return BudgetToReasoningEffort(budget), true
		}
		return "auto", true
	case "adaptive", "auto":
		outputConfig := toMap(root["output_config"])
		effort := strings.ToLower(strings.TrimSpace(toString(outputConfig["effort"])))
		switch effort {
		case "", "auto":
			return "auto", true
		case "low", "medium", "high":
			return effort, true
		case "max":
			return "xhigh", true
		default:
			return "auto", true
		}
	default:
		return "", false
	}
}

// BudgetToReasoningEffort maps a Claude thinking budget_tokens value to an
// OpenAI reasoning_effort string.
func BudgetToReasoningEffort(budget int) string {
	switch {
	case budget <= 0:
		return "none"
	case budget <= 512:
		return "minimal"
	case budget <= 1024:
		return "low"
	case budget <= 8192:
		return "medium"
	case budget <= 24576:
		return "high"
	default:
		return "xhigh"
	}
}

// normalizeOpenAIContent collapses a content parts array: single text part
// becomes a plain string, empty returns "", otherwise the array is passed
// through.
func normalizeOpenAIContent(contentParts []any) any {
	if len(contentParts) == 0 {
		return ""
	}

	if len(contentParts) == 1 {
		if part := toMap(contentParts[0]); part != nil && strings.EqualFold(toString(part["type"]), "text") {
			return toString(part["text"])
		}
	}

	return contentParts
}

// marshalJSONObject marshals a value as a JSON object string. Returns "{}"
// for nil, empty, or invalid input.
func marshalJSONObject(value any) string {
	switch typed := value.(type) {
	case nil:
		return "{}"
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return "{}"
		}
		if json.Valid([]byte(trimmed)) {
			return trimmed
		}
		encoded, _ := json.Marshal(trimmed)
		return string(encoded)
	default:
		encoded, err := json.Marshal(typed)
		if err != nil || len(encoded) == 0 {
			return "{}"
		}
		return string(encoded)
	}
}

// SanitizeClaudeToolID ensures a tool call ID contains only valid
// characters. Returns a generated ID if the input is empty or fully
// stripped.
func SanitizeClaudeToolID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return "toolu_" + randHex(13)
	}

	var builder strings.Builder
	for _, ch := range id {
		switch {
		case ch >= 'a' && ch <= 'z':
			builder.WriteRune(ch)
		case ch >= 'A' && ch <= 'Z':
			builder.WriteRune(ch)
		case ch >= '0' && ch <= '9':
			builder.WriteRune(ch)
		case ch == '_' || ch == '-':
			builder.WriteRune(ch)
		}
	}

	if builder.Len() == 0 {
		return "toolu_" + randHex(13)
	}
	return builder.String()
}

// --- Generic JSON helpers ---

func toMap(value any) map[string]any {
	typed, _ := value.(map[string]any)
	return typed
}

func toSlice(value any) []any {
	typed, _ := value.([]any)
	return typed
}

func toStringSlice(value any) []string {
	values := toSlice(value)
	if len(values) == 0 {
		return nil
	}

	out := make([]string, 0, len(values))
	for _, entry := range values {
		text := strings.TrimSpace(toString(entry))
		if text == "" {
			continue
		}
		out = append(out, text)
	}
	return out
}

func toString(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return ""
	}
}

func toBool(value any) bool {
	typed, _ := value.(bool)
	return typed
}

func toInt(value any) (int, bool) {
	switch typed := value.(type) {
	case float64:
		return int(typed), true
	case int:
		return typed, true
	case int64:
		return int(typed), true
	default:
		return 0, false
	}
}

func toFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func firstNonEmptyStr(values ...any) string {
	for _, value := range values {
		text := strings.TrimSpace(toString(value))
		if text != "" {
			return text
		}
	}
	return ""
}

// cloneMap returns a shallow copy of a map[string]any.
func cloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
