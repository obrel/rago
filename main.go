package main

import (
	"context"
	"flag"
	"io"
	"os"
	"time"

	"github.com/obrel/go-lib/pkg/log"
	"github.com/obrel/rago/pkg/backend"
	"github.com/obrel/rago/pkg/backend/openai"
	"github.com/obrel/rago/pkg/db"
	"github.com/obrel/rago/pkg/db/pgvector"
)

var (
	embedModel  *string
	genModel    *string
	databaseURL *string
)

func main() {
	embedModel = flag.String("embed", "", "Embedding model")
	genModel = flag.String("model", "", "Generation model")
	databaseURL = flag.String("dsn", "", "Database url")
	flag.Parse()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.For("example", "main").Fatal("API key not found in environment variables.")
	}

	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		log.For("example", "main").Fatal("Base URL not found in environment variables.")
	}

	embeddingBackend, err := backend.NewBackend("openai", []backend.Option{
		openai.ApiKey(apiKey),
		openai.BaseURL(baseURL),
		openai.Model(*embedModel),
	}...)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	generationBackend, err := backend.NewBackend("openai", []backend.Option{
		openai.ApiKey(apiKey),
		openai.BaseURL(baseURL),
		openai.Model(*genModel),
	}...)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	vectorDB, err := db.NewDB("pgvector", []db.Option{
		pgvector.DSN(*databaseURL),
	}...)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	defer vectorDB.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	file, err := os.Open("document.txt")
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	ragContent := string(content)
	query := "Kapan hari kemerdekaan Indonesia?"

	embedding, err := embeddingBackend.Embed(ctx, ragContent)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	_, err = vectorDB.InsertDocument(ctx, ragContent, embedding)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	queryEmbedding, err := embeddingBackend.Embed(ctx, query)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	retrievedDocs, err := vectorDB.QueryRelevantDocuments(ctx, queryEmbedding)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	augmentedQuery := db.CombineQueryWithContext(query, retrievedDocs)

	prompt := backend.NewPrompt().
		AddMessage("system", "Kamu adalah seorang asisten AI. Gunakan context yang tersedia untuk menjawab pertanyaan user seakurat mungkin.").
		AddMessage("user", augmentedQuery).
		SetParameters(backend.Parameters{
			MaxTokens:        150,
			Temperature:      0.7,
			TopP:             0.9,
			FrequencyPenalty: 0.5,
			PresencePenalty:  0.6,
		})

	response, err := generationBackend.Generate(ctx, prompt)
	if err != nil {
		log.For("example", "main").Fatal(err)
	}

	log.Printf("Output: %s", response)
}
