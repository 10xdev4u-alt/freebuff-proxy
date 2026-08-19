package convert

import (
	"encoding/json"
	"strings"
)

// ConvertResponsesToChatParams translates an OpenAI Responses API request
// body into an OpenAI chat-completions request body. Returns the converted
// payload, model name, stream flag, and any error.
func ConvertResponsesToChatParams(body []byte) (map[string]any, string, bool, error) {
	var params map[string]any
	if err := json.Unmarshal(body, &params); err != nil {
		return nil, "", false, nil
	}

	stream := toBool(params["stream"])
	modelName := strings.TrimSpace(toString(params["model"]))
	if modelName == "" {
		return nil, "", false, nil
	}

	chat := make(map[string]any)

	// Pass through fields that exist in both formats.
	for _, key := range []string{
		"temperature", "top_p", "tools", "tool_choice",
		"parallel_tool_calls", "stop", "seed", "store",
		"metadata", "user", "stream",
	} {
		if v, ok := params[key]; ok && v != nil {
			chat[key] = v
		}
	}

	// max_output_tokens → max_completion_tokens
	if maxOutput, ok := toInt(params["max_output_tokens"]); ok && maxOutput > 0 {
		chat["max_completion_tokens"] = maxOutput
	}

	// reasoning.effort → reasoning_effort
	if reasoning, ok := params["reasoning"].(map[string]any); ok {
		if effort, ok := reasoning["effort"].(string); ok && effort != "" {
			chat["reasoning_effort"] = effort
		}
	}

	// text.format → response_format (skip "text" type which is the default)
	if text, ok := params["text"].(map[string]any); ok {
		if format, ok := text["format"].(map[string]any); ok {
			formatType, _ := format["type"].(string)
			if formatType != "" && formatType != "text" {
				rf := map[string]any{"type": formatType}
				if jsonSchema, ok := format["json_schema"]; ok {
					rf["json_schema"] = jsonSchema
				}
				chat["response_format"] = rf
			}
		}
	}

	// Tools: filter to function type only, wrap in {type: "function", function: {...}}
	if rawTools, ok := params["tools"].([]any); ok && len(rawTools) > 0 {
		chatTools := make([]any, 0, len(rawTools))
		for _, rawTool := range rawTools {
			tool, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			toolType, _ := tool["type"].(string)
			if toolType != "function" {
				continue
			}
			chatTools = append(chatTools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        toString(tool["name"]),
					"description": toString(tool["description"]),
					"parameters":  coalesce(tool["parameters"], map[string]any{"type": "object", "properties": map[string]any{}}),
				},
			})
		}
		if len(chatTools) > 0 {
			chat["tools"] = chatTools
		}
	}

	// tool_choice: function type → {type: "function", function: {name}}, else "auto"
	if tc, ok := params["tool_choice"].(map[string]any); ok {
		tcType, _ := tc["type"].(string)
		if tcType == "function" {
			if name, ok := tc["name"].(string); ok && name != "" {
				chat["tool_choice"] = map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": name,
					},
				}
			} else {
				chat["tool_choice"] = "auto"
			}
		} else {
			chat["tool_choice"] = "auto"
		}
	}

	chat["model"] = modelName
	chat["messages"] = ResponsesInputToMessages(params["input"], toString(params["instructions"]))

	return chat, modelName, stream, nil
}

// ResponsesInputToMessages converts a Responses API input field (string or
// array of parts) into OpenAI chat messages. An instructions string is
// prepended as a system message.
func ResponsesInputToMessages(input any, instructions string) []any {
	messages := make([]any, 0, 8)

	if instructions != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": instructions,
		})
	}

	switch typed := input.(type) {
	case string:
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": typed,
		})
		return messages

	case []any:
		for _, rawItem := range typed {
			item, ok := rawItem.(map[string]any)
			if !ok {
				continue
			}

			itemType, _ := item["type"].(string)

			// function_call_output → tool message
			if itemType == "function_call_output" {
				callID := toString(item["call_id"])
				output := item["output"]
				var outputStr string
				if s, ok := output.(string); ok {
					outputStr = s
				} else if output != nil {
					b, _ := json.Marshal(output)
					outputStr = string(b)
				}
				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      outputStr,
				})
				continue
			}

			// Skip types we can't handle locally
			if itemType == "function_call" || itemType == "reasoning" || itemType == "item_reference" {
				continue
			}

			role := toString(item["role"])
			if role == "" {
				role = "user"
			}

			content := item["content"]
			switch c := content.(type) {
			case string:
				messages = append(messages, map[string]any{
					"role":    role,
					"content": c,
				})
			case []any:
				parts := make([]any, 0, len(c))
				for _, rawPart := range c {
					part, ok := rawPart.(map[string]any)
					if !ok {
						continue
					}
					partType, _ := part["type"].(string)
					switch partType {
					case "input_text", "output_text":
						parts = append(parts, map[string]any{
							"type": "text",
							"text": toString(part["text"]),
						})
					case "text":
						if text, ok := part["text"].(string); ok {
							parts = append(parts, part)
							_ = text // already appended
						}
					}
				}
				if len(parts) > 0 {
					messages = append(messages, map[string]any{
						"role":    role,
						"content": parts,
					})
				} else {
					messages = append(messages, map[string]any{
						"role":    role,
						"content": "",
					})
				}
			default:
				messages = append(messages, map[string]any{
					"role":    role,
					"content": "",
				})
			}
		}
		return messages

	default:
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": "",
		})
		return messages
	}
}

// coalesce returns the first non-nil value.
func coalesce(values ...any) any {
	for _, v := range values {
		if v != nil {
			return v
		}
	}
	return nil
}
