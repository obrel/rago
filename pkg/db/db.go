package db

import (
	"context"
	"fmt"
	"sync"

	"github.com/obrel/go-lib/pkg/log"
)

var (
	dbs  = map[string]Factory{}
	lock sync.RWMutex
)

type Document struct {
	ID       string
	Metadata map[string]interface{}
}

type Option interface{}

type DB interface {
	InsertDocument(ctx context.Context, content string, embedding []float32) (string, error)
	SaveEmbeddings(ctx context.Context, docID string, embedding []float32, metadata map[string]interface{}) error
	QueryRelevantDocuments(ctx context.Context, embedding []float32) ([]Document, error)
	FindByDocID(ctx context.Context, docID string) (Document, error)
	UpdateDocument(ctx context.Context, content string, embedding []float32) (string, error)
	DeleteEmbeddings(ctx context.Context, docID string) error
	Close()
}

type Factory func(...Option) (DB, error)

func NewDB(db string, opts ...Option) (DB, error) {
	lock.RLock()
	defer lock.RUnlock()

	s, ok := dbs[db]
	if !ok {
		log.For("db", "new").Fatal("Database not found.")
	}

	return s(opts...)
}

func Register(db string, d Factory) {
	lock.Lock()
	defer lock.Unlock()

	if d == nil {
		log.For("db", "register").Fatal("Invalid database.")
	}

	if _, ok := dbs[db]; ok {
		log.For("db", "register").Fatalf("Database %s already registered.", db)
	}

	dbs[db] = d
}

func CombineQueryWithContext(query string, retrievedDocs []Document) string {
	var context string

	for _, doc := range retrievedDocs {
		context += fmt.Sprintf("%s\n", doc.Metadata["content"])
	}

	return fmt.Sprintf("Context: %s\n\nQuery: %s", context, query)
}
