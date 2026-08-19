package convert

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

var (
	clDataPrefix = []byte("data:")
	clDoneToken  = []byte("[DONE]")
	clColon      = []byte(":")
)

// --- OpenAI response types for Claude conversion ---

type openAIChatCompletion struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	FinishReason string        `json:"finish_reason"`
	Message      openAIMessage `json:"message"`
	Delta        openAIDelta   `json:"delta"`
}

type openAIMessage struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
	ReasoningContent json.RawMessage  `json:"reasoning_content,omitempty"`
}

type openAIDelta struct {
	Role             string                 `json:"role"`
	Content          string                 `json:"content"`
	ToolCalls        []openAIStreamToolCall `json:"tool_calls,omitempty"`
	ReasoningContent json.RawMessage        `json:"reasoning_content,omitempty"`
}

type openAIToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

type openAIStreamToolCall struct {
	Index    int            `json:"index"`
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIUsage struct {
	PromptTokens        int64                     `json:"prompt_tokens"`
	CompletionTokens    int64                     `json:"completion_tokens"`
	TotalTokens         int64                     `json:"total_tokens"`
	PromptTokensDetails *openAIPromptTokensDetail `json:"prompt_tokens_details,omitempty"`
}

type openAIPromptTokensDetail struct {
	CachedTokens int64 `json:"cached_tokens"`
}

// --- Claude SSE event types ---

// ClaudeSSEEvent represents a single Claude SSE event to write.
type ClaudeSSEEvent struct {
	Name    string
	Payload []byte
}

// ClaudeStreamState tracks the state of a streaming conversion from OpenAI
// SSE chunks to Claude SSE events.
type ClaudeStreamState struct {
	MessageID            string
	Model                string
	MessageStarted       bool
	TextContentStarted   bool
	TextContentBlockIdx  int
	ThinkingStarted      bool
	ThinkingBlockIdx     int
	NextBlockIdx         int
	ToolBlocks           map[int]*ClaudeToolCallState
	ToolBlockIndexes     map[int]int
	FinishReason         string
	SawToolCall          bool
	MessageDeltaSent     bool
	MessageStopSent      bool
	ContentBlocksStopped bool
}

// ClaudeToolCallState accumulates streaming tool call deltas.
type ClaudeToolCallState struct {
	ID        string
	Name      string
	Started   bool
	Arguments strings.Builder
}

// NewClaudeStreamState creates a fresh streaming state for the given model.
func NewClaudeStreamState(model string) *ClaudeStreamState {
	return &ClaudeStreamState{
		Model:           model,
		TextContentBlockIdx: -1,
		ThinkingBlockIdx:    -1,
		ToolBlocks:          make(map[int]*ClaudeToolCallState),
		ToolBlockIndexes:    make(map[int]int),
	}
}

// WriteClaudeSuccessResponse dispatches to streaming or non-streaming
// response writing based on the stream flag.
func WriteClaudeSuccessResponse(w http.ResponseWriter, resp *http.Response, requestedModel string, stream bool) error {
	if stream {
		return WriteClaudeStreamingResponse(w, resp, requestedModel)
	}
	return WriteClaudeNonStreamResponse(w, resp)
}

// WriteClaudeNonStreamResponse reads the full upstream OpenAI response body,
// converts it to Claude format, and writes it to w.
func WriteClaudeNonStreamResponse(w http.ResponseWriter, resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	converted, err := convertOpenAINonStreamResponseToClaude(body)
	if err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, err = w.Write(converted)
	return err
}

// convertOpenAINonStreamResponseToClaude transforms a full OpenAI
// chat-completion JSON response into a Claude Messages API response.
func convertOpenAINonStreamResponseToClaude(body []byte) ([]byte, error) {
	var response openAIChatCompletion
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode upstream response: %w", err)
	}

	message := map[string]any{
		"id":            response.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         response.Model,
		"content":       []any{},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}

	hasToolCall := false
	if len(response.Choices) > 0 {
		choice := response.Choices[0]
		for _, text := range collectReasoningTexts(choice.Message.ReasoningContent) {
			message["content"] = append(message["content"].([]any), map[string]any{
				"type":     "thinking",
				"thinking": text,
			})
		}
		for _, block := range convertOpenAIContentToClaudeBlocks(choice.Message.Content) {
			message["content"] = append(message["content"].([]any), block)
		}
		for _, toolCall := range choice.Message.ToolCalls {
			hasToolCall = true
			message["content"] = append(message["content"].([]any), map[string]any{
				"type":  "tool_use",
				"id":    SanitizeClaudeToolID(toolCall.ID),
				"name":  toolCall.Function.Name,
				"input": parseJSONObject(toolCall.Function.Arguments),
			})
		}
		if choice.FinishReason != "" {
			message["stop_reason"] = mapOpenAIFinishReasonToClaude(choice.FinishReason)
		}
	}

	if response.Usage != nil {
		inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(response.Usage)
		usage := message["usage"].(map[string]any)
		usage["input_tokens"] = inputTokens
		usage["output_tokens"] = outputTokens
		if cachedTokens > 0 {
			usage["cache_read_input_tokens"] = cachedTokens
		}
	}

	if message["stop_reason"] == "end_turn" && hasToolCall {
		message["stop_reason"] = "tool_use"
	}

	return json.Marshal(message)
}

// WriteClaudeStreamingResponse reads upstream OpenAI SSE chunks and
// converts them to Claude SSE events, writing them to w.
func WriteClaudeStreamingResponse(w http.ResponseWriter, resp *http.Response, requestedModel string) error {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(resp.StatusCode)

	return convertAndRelayStream(w, bufio.NewReader(resp.Body), requestedModel)
}

// WriteClaudeStreamingResponseFromReader reads an OpenAI SSE stream from
// an io.Reader and writes Claude SSE events to w. Unlike
// WriteClaudeStreamingResponse, it does not set headers or write a status
// code, allowing the caller to control those.
func WriteClaudeStreamingResponseFromReader(w http.ResponseWriter, r io.Reader, requestedModel string) error {
	return convertAndRelayStream(w, bufio.NewReader(r), requestedModel)
}

// convertAndRelayStream is the shared implementation for both streaming
// response writers. It reads SSE lines from reader, converts OpenAI chunks
// to Claude events, and writes them to w.
func convertAndRelayStream(w http.ResponseWriter, reader *bufio.Reader, requestedModel string) error {
	flusher, _ := w.(http.Flusher)
	state := NewClaudeStreamState(requestedModel)

	sawDone := false
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			trimmed := bytes.TrimSpace(line)
			if len(trimmed) > 0 && !bytes.HasPrefix(trimmed, clColon) && bytes.HasPrefix(trimmed, clDataPrefix) {
				payload := bytes.TrimSpace(trimmed[5:])
				if bytes.Equal(payload, clDoneToken) {
					sawDone = true
				}

				events, convErr := ConvertOpenAIStreamPayloadToClaudeEvents(payload, state)
				if convErr != nil {
					return convErr
				}
				if err := WriteClaudeSSEEvents(w, events); err != nil {
					return err
				}
				if flusher != nil && len(events) > 0 {
					flusher.Flush()
				}
			}
		}

		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}

	if !sawDone {
		events, err := FinalizeClaudeStream(state)
		if err != nil {
			return err
		}
		if err := WriteClaudeSSEEvents(w, events); err != nil {
			return err
		}
		if flusher != nil && len(events) > 0 {
			flusher.Flush()
		}
	}

	return nil
}

// WriteClaudeNonStreamResponseFromReader reads a full OpenAI chat-completion
// response from an io.Reader, converts it to Claude format, and writes it
// to w. Unlike WriteClaudeNonStreamResponse, it does not set headers or
// write a status code.
func WriteClaudeNonStreamResponseFromReader(w http.ResponseWriter, r io.Reader) error {
	body, err := io.ReadAll(r)
	if err != nil {
		return err
	}

	converted, err := convertOpenAINonStreamResponseToClaude(body)
	if err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/json")
	_, err = w.Write(converted)
	return err
}

// ConvertOpenAIStreamPayloadToClaudeEvents converts a single OpenAI SSE
// data payload (or [DONE]) into zero or more Claude SSE events.
func ConvertOpenAIStreamPayloadToClaudeEvents(payload []byte, state *ClaudeStreamState) ([]ClaudeSSEEvent, error) {
	if bytes.Equal(bytes.TrimSpace(payload), clDoneToken) {
		return FinalizeClaudeStream(state)
	}

	var chunk openAIChatCompletion
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return nil, fmt.Errorf("decode upstream stream chunk: %w", err)
	}

	events := make([]ClaudeSSEEvent, 0, 8)
	if state.MessageID == "" {
		state.MessageID = chunk.ID
	}
	if state.MessageID == "" {
		state.MessageID = "msg_" + randHex(13)
	}
	if strings.TrimSpace(chunk.Model) != "" {
		state.Model = chunk.Model
	}

	if !state.MessageStarted {
		messageStartPayload, err := json.Marshal(map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            state.MessageID,
				"type":          "message",
				"role":          "assistant",
				"model":         state.Model,
				"content":       []any{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage": map[string]any{
					"input_tokens":  0,
					"output_tokens": 0,
				},
			},
		})
		if err != nil {
			return nil, err
		}
		events = append(events, ClaudeSSEEvent{Name: "message_start", Payload: messageStartPayload})
		state.MessageStarted = true
	}

	if len(chunk.Choices) > 0 {
		choice := chunk.Choices[0]
		for _, text := range collectReasoningTexts(choice.Delta.ReasoningContent) {
			stopTextContentBlock(state, &events)
			if !state.ThinkingStarted {
				index := nextClaudeBlockIndex(state, &state.ThinkingBlockIdx)
				payload, err := json.Marshal(map[string]any{
					"type":  "content_block_start",
					"index": index,
					"content_block": map[string]any{
						"type":     "thinking",
						"thinking": "",
					},
				})
				if err != nil {
					return nil, err
				}
				events = append(events, ClaudeSSEEvent{Name: "content_block_start", Payload: payload})
				state.ThinkingStarted = true
			}

			payload, err := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": state.ThinkingBlockIdx,
				"delta": map[string]any{
					"type":     "thinking_delta",
					"thinking": text,
				},
			})
			if err != nil {
				return nil, err
			}
			events = append(events, ClaudeSSEEvent{Name: "content_block_delta", Payload: payload})
		}

		if choice.Delta.Content != "" {
			stopThinkingContentBlock(state, &events)
			if !state.TextContentStarted {
				index := nextClaudeBlockIndex(state, &state.TextContentBlockIdx)
				payload, err := json.Marshal(map[string]any{
					"type":  "content_block_start",
					"index": index,
					"content_block": map[string]any{
						"type": "text",
						"text": "",
					},
				})
				if err != nil {
					return nil, err
				}
				events = append(events, ClaudeSSEEvent{Name: "content_block_start", Payload: payload})
				state.TextContentStarted = true
			}

			payload, err := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": state.TextContentBlockIdx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": choice.Delta.Content,
				},
			})
			if err != nil {
				return nil, err
			}
			events = append(events, ClaudeSSEEvent{Name: "content_block_delta", Payload: payload})
		}

		for _, toolCall := range choice.Delta.ToolCalls {
			state.SawToolCall = true
			stopThinkingContentBlock(state, &events)
			stopTextContentBlock(state, &events)

			accumulator := state.ToolBlocks[toolCall.Index]
			if accumulator == nil {
				accumulator = &ClaudeToolCallState{}
				state.ToolBlocks[toolCall.Index] = accumulator
			}

			if strings.TrimSpace(toolCall.ID) != "" {
				accumulator.ID = toolCall.ID
			}
			if strings.TrimSpace(toolCall.Function.Name) != "" {
				accumulator.Name = toolCall.Function.Name
			}
			if toolCall.Function.Arguments != "" {
				accumulator.Arguments.WriteString(toolCall.Function.Arguments)
			}

			if !accumulator.Started && accumulator.Name != "" {
				blockIndex := toolCallBlockIndex(state, toolCall.Index)
				payload, err := json.Marshal(map[string]any{
					"type":  "content_block_start",
					"index": blockIndex,
					"content_block": map[string]any{
						"type":  "tool_use",
						"id":    SanitizeClaudeToolID(accumulator.ID),
						"name":  accumulator.Name,
						"input": map[string]any{},
					},
				})
				if err != nil {
					return nil, err
				}
				events = append(events, ClaudeSSEEvent{Name: "content_block_start", Payload: payload})
				accumulator.Started = true
			}
		}

		if choice.FinishReason != "" {
			state.FinishReason = choice.FinishReason
			var err error
			events, err = appendClaudeFinalContentEvents(events, state)
			if err != nil {
				return nil, err
			}
		}
	}

	if state.FinishReason != "" && chunk.Usage != nil {
		var err error
		events, err = appendClaudeMessageDeltaAndStop(events, state, chunk.Usage)
		if err != nil {
			return nil, err
		}
	}

	return events, nil
}

// FinalizeClaudeStream emits any remaining content block stops, message
// delta, and message stop events for a stream that ended without [DONE].
func FinalizeClaudeStream(state *ClaudeStreamState) ([]ClaudeSSEEvent, error) {
	events := make([]ClaudeSSEEvent, 0, 6)
	var err error
	events, err = appendClaudeFinalContentEvents(events, state)
	if err != nil {
		return nil, err
	}
	return appendClaudeMessageDeltaAndStop(events, state, nil)
}

func appendClaudeFinalContentEvents(events []ClaudeSSEEvent, state *ClaudeStreamState) ([]ClaudeSSEEvent, error) {
	if state.ContentBlocksStopped {
		return events, nil
	}

	stopThinkingContentBlock(state, &events)
	stopTextContentBlock(state, &events)

	indexes := make([]int, 0, len(state.ToolBlocks))
	for index := range state.ToolBlocks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	for _, index := range indexes {
		toolCall := state.ToolBlocks[index]
		if !toolCall.Started {
			continue
		}

		blockIndex := toolCallBlockIndex(state, index)
		partialJSON := strings.TrimSpace(toolCall.Arguments.String())
		if partialJSON != "" {
			payload, err := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": blockIndex,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": partialJSON,
				},
			})
			if err != nil {
				return nil, err
			}
			events = append(events, ClaudeSSEEvent{Name: "content_block_delta", Payload: payload})
		}

		stopPayload, err := json.Marshal(map[string]any{
			"type":  "content_block_stop",
			"index": blockIndex,
		})
		if err != nil {
			return nil, err
		}
		events = append(events, ClaudeSSEEvent{Name: "content_block_stop", Payload: stopPayload})
	}

	state.ContentBlocksStopped = true
	return events, nil
}

func appendClaudeMessageDeltaAndStop(events []ClaudeSSEEvent, state *ClaudeStreamState, usage *openAIUsage) ([]ClaudeSSEEvent, error) {
	if !state.MessageDeltaSent {
		inputTokens, outputTokens, cachedTokens := int64(0), int64(0), int64(0)
		if usage != nil {
			inputTokens, outputTokens, cachedTokens = extractOpenAIUsage(usage)
		}

		usagePayload := map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		if cachedTokens > 0 {
			usagePayload["cache_read_input_tokens"] = cachedTokens
		}

		payload, err := json.Marshal(map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   mapOpenAIFinishReasonToClaude(effectiveOpenAIFinishReason(state)),
				"stop_sequence": nil,
			},
			"usage": usagePayload,
		})
		if err != nil {
			return nil, err
		}
		events = append(events, ClaudeSSEEvent{Name: "message_delta", Payload: payload})
		state.MessageDeltaSent = true
	}

	if !state.MessageStopSent {
		payload, err := json.Marshal(map[string]any{"type": "message_stop"})
		if err != nil {
			return nil, err
		}
		events = append(events, ClaudeSSEEvent{Name: "message_stop", Payload: payload})
		state.MessageStopSent = true
	}

	return events, nil
}

// --- Stream helper functions ---

func effectiveOpenAIFinishReason(state *ClaudeStreamState) string {
	if state.SawToolCall {
		return "tool_calls"
	}
	if strings.TrimSpace(state.FinishReason) == "" {
		return "stop"
	}
	return state.FinishReason
}

func nextClaudeBlockIndex(state *ClaudeStreamState, slot *int) int {
	if *slot >= 0 {
		return *slot
	}
	index := state.NextBlockIdx
	state.NextBlockIdx++
	*slot = index
	return index
}

func toolCallBlockIndex(state *ClaudeStreamState, toolIndex int) int {
	if index, ok := state.ToolBlockIndexes[toolIndex]; ok {
		return index
	}
	index := state.NextBlockIdx
	state.NextBlockIdx++
	state.ToolBlockIndexes[toolIndex] = index
	return index
}

func stopThinkingContentBlock(state *ClaudeStreamState, events *[]ClaudeSSEEvent) {
	if !state.ThinkingStarted {
		return
	}
	payload, err := json.Marshal(map[string]any{
		"type":  "content_block_stop",
		"index": state.ThinkingBlockIdx,
	})
	if err == nil {
		*events = append(*events, ClaudeSSEEvent{Name: "content_block_stop", Payload: payload})
	}
	state.ThinkingStarted = false
	state.ThinkingBlockIdx = -1
}

func stopTextContentBlock(state *ClaudeStreamState, events *[]ClaudeSSEEvent) {
	if !state.TextContentStarted {
		return
	}
	payload, err := json.Marshal(map[string]any{
		"type":  "content_block_stop",
		"index": state.TextContentBlockIdx,
	})
	if err == nil {
		*events = append(*events, ClaudeSSEEvent{Name: "content_block_stop", Payload: payload})
	}
	state.TextContentStarted = false
	state.TextContentBlockIdx = -1
}

// WriteClaudeSSEEvents writes a sequence of Claude SSE events to w.
func WriteClaudeSSEEvents(w http.ResponseWriter, events []ClaudeSSEEvent) error {
	for _, event := range events {
		if _, err := io.WriteString(w, "event: "+event.Name+"\n"); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "data: "); err != nil {
			return err
		}
		if _, err := w.Write(event.Payload); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "\n\n"); err != nil {
			return err
		}
	}
	return nil
}

// --- Response conversion helpers ---

func convertOpenAIContentToClaudeBlocks(raw json.RawMessage) []any {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}

	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err == nil && strings.TrimSpace(text) != "" {
			return []any{map[string]any{"type": "text", "text": text}}
		}
		return nil
	}

	if raw[0] != '[' {
		return nil
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil
	}

	blocks := make([]any, 0, len(items))
	for _, item := range items {
		switch strings.ToLower(strings.TrimSpace(toString(item["type"]))) {
		case "text":
			text := toString(item["text"])
			if strings.TrimSpace(text) == "" {
				continue
			}
			blocks = append(blocks, map[string]any{"type": "text", "text": text})
		case "reasoning":
			text := strings.TrimSpace(firstNonEmptyStr(item["text"], item["thinking"]))
			if text == "" {
				continue
			}
			blocks = append(blocks, map[string]any{"type": "thinking", "thinking": text})
		case "tool_calls":
			for _, rawToolCall := range toSlice(item["tool_calls"]) {
				toolCall := toMap(rawToolCall)
				if toolCall == nil {
					continue
				}
				function := toMap(toolCall["function"])
				blocks = append(blocks, map[string]any{
					"type":  "tool_use",
					"id":    SanitizeClaudeToolID(toString(toolCall["id"])),
					"name":  toString(function["name"]),
					"input": parseJSONObject(toString(function["arguments"])),
				})
			}
		}
	}

	return blocks
}

func collectReasoningTexts(raw json.RawMessage) []string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil
	}

	texts := make([]string, 0)
	collectReasoningTextValues(value, &texts)
	return texts
}

func collectReasoningTextValues(value any, out *[]string) {
	switch typed := value.(type) {
	case string:
		if strings.TrimSpace(typed) != "" {
			*out = append(*out, typed)
		}
	case []any:
		for _, item := range typed {
			collectReasoningTextValues(item, out)
		}
	case map[string]any:
		if text := strings.TrimSpace(firstNonEmptyStr(typed["text"], typed["thinking"])); text != "" {
			*out = append(*out, text)
		}
	}
}

func parseJSONObject(raw string) map[string]any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}
	}

	var value any
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return map[string]any{}
	}

	if object, ok := value.(map[string]any); ok {
		return object
	}
	return map[string]any{}
}

func extractOpenAIUsage(usage *openAIUsage) (int64, int64, int64) {
	if usage == nil {
		return 0, 0, 0
	}

	inputTokens := usage.PromptTokens
	outputTokens := usage.CompletionTokens
	cachedTokens := int64(0)
	if usage.PromptTokensDetails != nil {
		cachedTokens = usage.PromptTokensDetails.CachedTokens
	}

	if cachedTokens > 0 {
		if inputTokens >= cachedTokens {
			inputTokens -= cachedTokens
		} else {
			inputTokens = 0
		}
	}

	return inputTokens, outputTokens, cachedTokens
}

func mapOpenAIFinishReasonToClaude(reason string) string {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case "tool_calls", "function_call":
		return "tool_use"
	case "length":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

// WriteClaudePassthroughError writes an upstream error body in Claude error
// format.
func WriteClaudePassthroughError(w http.ResponseWriter, statusCode int, body []byte) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) > 0 && json.Valid(trimmed) {
		message, errorType, _ := extractUpstreamError(trimmed)
		WriteClaudeError(w, statusCode, message, normalizeClaudeErrorType(statusCode, errorType))
		return
	}
	WriteClaudeError(w, statusCode, strings.TrimSpace(string(trimmed)), normalizeClaudeErrorType(statusCode, ""))
}

// WriteClaudeError writes a Claude-formatted error response.
func WriteClaudeError(w http.ResponseWriter, statusCode int, message, errorType string) {
	if strings.TrimSpace(message) == "" {
		message = http.StatusText(statusCode)
	}
	if strings.TrimSpace(errorType) == "" {
		errorType = normalizeClaudeErrorType(statusCode, "")
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	enc := json.NewEncoder(w)
	_ = enc.Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    errorType,
			"message": message,
		},
	})
}

func normalizeClaudeErrorType(statusCode int, upstreamType string) string {
	switch strings.TrimSpace(upstreamType) {
	case "invalid_request_error", "authentication_error", "permission_error",
		"not_found_error", "rate_limit_error", "api_error", "overloaded_error":
		return upstreamType
	}

	switch statusCode {
	case http.StatusBadRequest, http.StatusMethodNotAllowed, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, http.StatusBadGateway, http.StatusGatewayTimeout:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// extractUpstreamError parses a JSON error body to pull out message and type.
func extractUpstreamError(body []byte) (message, errorType string, ok bool) {
	var envelope struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
		Message string `json:"message"`
		Status  string `json:"status"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return "", "", false
	}
	if envelope.Error != nil {
		return envelope.Error.Message, envelope.Error.Type, true
	}
	if envelope.Message != "" {
		return envelope.Message, envelope.Status, true
	}
	return "", "", false
}
