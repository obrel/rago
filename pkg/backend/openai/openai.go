package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/obrel/rago/pkg/backend"
)

var (
	defaultTimeout = 30 * time.Second
	defaultBaseURL = "https://api.openai.com"
)

type OpenAI struct {
	apiKey     string
	baseURL    string
	model      string
	HTTPClient *http.Client
}

type OpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type OpenAIEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func (o *OpenAI) Generate(ctx context.Context, prompt *backend.Prompt) (string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	url := o.baseURL + "/v1/chat/completions"
	reqBody := map[string]interface{}{
		"model":             o.model,
		"messages":          prompt.Messages,
		"max_tokens":        prompt.Parameters.MaxTokens,
		"temperature":       prompt.Parameters.Temperature,
		"top_p":             prompt.Parameters.TopP,
		"frequency_penalty": prompt.Parameters.FrequencyPenalty,
		"presence_penalty":  prompt.Parameters.PresencePenalty,
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("Failed to marshal request body: %v.", err)
	}

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return "", fmt.Errorf("Failed to create request: %v.", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %v.", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("Failed to read response body: %v.", err)
		}

		return "", fmt.Errorf("Failed to generate response from OpenAI: [%d] %v.", resp.StatusCode, string(bodyBytes))
	}

	var result OpenAIResponse

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("Failed to decode response: %v.", err)
	}

	return result.Choices[0].Message.Content, nil
}

func (o *OpenAI) Embed(ctx context.Context, text string) ([]float32, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	url := o.baseURL + "/v1/embeddings"
	reqBody := map[string]interface{}{
		"model": o.model,
		"input": text,
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("Failed to marshal request body: %v.", err)
	}

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("Failed to create request: %v.", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %v.", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("Failed to read response body: %w", err)
		}

		return nil, fmt.Errorf("Failed to generate embedding from OpenAI: [%d] %v.", resp.StatusCode, string(bodyBytes))
	}

	var result OpenAIEmbeddingResponse

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("Failed to decode response: %v.", err)
	}

	return result.Data[0].Embedding, nil
}

func ApiKey(s string) backend.Option {
	return func(o *OpenAI) {
		o.apiKey = s
	}
}

func BaseURL(s string) backend.Option {
	return func(o *OpenAI) {
		o.baseURL = s
	}
}

func Model(s string) backend.Option {
	return func(o *OpenAI) {
		o.model = s
	}
}

func init() {
	backend.Register("openai", func(opts ...backend.Option) (backend.Backend, error) {
		b := &OpenAI{}

		for _, opt := range opts {
			switch f := opt.(type) {
			case func(*OpenAI):
				f(b)
			default:
				return nil, fmt.Errorf("Unknown option.")
			}
		}

		b.HTTPClient = &http.Client{
			Timeout: defaultTimeout,
		}

		return b, nil
	})
}
