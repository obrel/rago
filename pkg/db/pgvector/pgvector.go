package pgvector

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/obrel/rago/pkg/db"
	"github.com/pgvector/pgvector-go"
)

type PGVector struct {
	dsn  string
	conn *pgxpool.Pool
}

func (p *PGVector) InsertDocument(ctx context.Context, content string, embedding []float32) (string, error) {
	docID := uuid.New().String()
	metadata := map[string]interface{}{
		"content": content,
	}

	err := p.SaveEmbeddings(ctx, docID, embedding, metadata)
	if err != nil {
		return "", err
	}

	return docID, nil
}

func (p *PGVector) SaveEmbeddings(ctx context.Context, docID string, embedding []float32, metadata map[string]interface{}) error {
	vector := pgvector.NewVector(embedding)
	query := `INSERT INTO openai_embeddings (doc_id, embedding, metadata) VALUES ($1, $2, $3)`

	_, err := p.conn.Exec(ctx, query, docID, vector, metadata)
	if err != nil {
		return fmt.Errorf("Failed to insert document: %v.", err)
	}

	return nil
}

func (p *PGVector) QueryRelevantDocuments(ctx context.Context, embedding []float32) ([]db.Document, error) {
	vector := pgvector.NewVector(embedding)
	query := `SELECT doc_id, metadata FROM openai_embeddings ORDER BY embedding <-> $1 LIMIT 5`

	rows, err := p.conn.Query(ctx, query, vector)
	if err != nil {
		return nil, fmt.Errorf("Failed to query relevant documents: %v.", err)
	}

	defer rows.Close()

	var docs []db.Document

	for rows.Next() {
		var doc db.Document

		if err := rows.Scan(&doc.ID, &doc.Metadata); err != nil {
			return nil, fmt.Errorf("Failed to scan row: %w", err)
		}

		docs = append(docs, doc)
	}

	return docs, nil
}

func (p *PGVector) FindByDocID(ctx context.Context, docID string) (db.Document, error) {
	query := `SELECT doc_id, metadata FROM openai_embeddings WHERE doc_id $1`
	row := p.conn.QueryRow(ctx, query, docID)
	doc := db.Document{}

	if err := row.Scan(&doc.ID, &doc.Metadata); err != nil {
		return doc, fmt.Errorf("Failed to scan row: %w", err)
	}

	return doc, nil
}

func (p *PGVector) UpdateDocument(ctx context.Context, docID, content string, embedding []float32) (string, error) {
	metadata := map[string]interface{}{
		"content": content,
	}

	err := p.DeleteEmbeddings(ctx, docID)
	if err != nil {
		return "", err
	}

	err = p.SaveEmbeddings(ctx, docID, embedding, metadata)
	if err != nil {
		return "", err
	}

	return docID, nil
}

func (p *PGVector) DeleteEmbeddings(ctx context.Context, docID string) error {
	query := `DELETE FROM openai_embeddings WHERE doc_id = $1`

	_, err := p.conn.Exec(ctx, query, docID)
	if err != nil {
		return fmt.Errorf("Failed to delete document: %v.", err)
	}

	return nil
}

func (p *PGVector) Close() {
	p.conn.Close()
}

func DSN(s string) db.Option {
	return func(d *PGVector) {
		d.dsn = s
	}
}

func PGXPool(p *pgxpool.Pool) db.Option {
	return func(d *PGVector) {
		d.conn = p
	}
}

func init() {
	db.Register("pgvector", func(opts ...db.Option) (db.DB, error) {
		d := &PGVector{}

		for _, opt := range opts {
			switch f := opt.(type) {
			case func(*PGVector):
				f(d)
			default:
				return nil, fmt.Errorf("Unknown option.")
			}
		}

		if d.dsn != "" {
			pool, err := pgxpool.Connect(context.Background(), d.dsn)
			if err != nil {
				return nil, fmt.Errorf("Failed to connect to database: %v.", err)
			}

			d.conn = pool
		} else if d.conn == nil {
			return nil, fmt.Errorf("Database connection not found.")
		}

		return d, nil
	})
}
