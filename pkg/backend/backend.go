package backend

import (
	"context"
	"sync"

	"github.com/obrel/go-lib/pkg/log"
)

var (
	backends = map[string]Factory{}
	lock     sync.RWMutex
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Parameters struct {
	MaxTokens        int     `json:"max_tokens"`
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"top_p"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`
}

type Option interface{}

type Backend interface {
	Generate(ctx context.Context, prompt *Prompt) (string, error)
	Embed(ctx context.Context, input string) ([]float32, error)
}

type Factory func(...Option) (Backend, error)

func NewBackend(bc string, opts ...Option) (Backend, error) {
	lock.RLock()
	defer lock.RUnlock()

	b, ok := backends[bc]
	if !ok {
		log.For("backend", "new").Fatal("Backend not found.")
	}

	return b(opts...)
}

func Register(bc string, b Factory) {
	lock.Lock()
	defer lock.Unlock()

	if b == nil {
		log.For("backend", "register").Fatal("Invalid backend.")
	}

	if _, ok := backends[bc]; ok {
		log.For("backend", "register").Fatalf("Backend %s already registered.", bc)
	}

	backends[bc] = b
}
