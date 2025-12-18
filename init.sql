-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create index for faster similarity search (optional, add after data)
-- CREATE INDEX ON photos USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
