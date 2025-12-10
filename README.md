# Tree Identification & Health Assessment API

FastAPI backend for combined tree re-identification and health assessment.

## Features

- Tree Re-ID using ConvNeXt embeddings (1024-dim)
- Health assessment from tree photos
- Gallery management with **pgvector** similarity search
- PostgreSQL + pgvector for native vector operations
- Dockerized for easy deployment
- CORS enabled for mobile app access

## Quick Start (Docker)

```bash
# Start everything
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop
docker-compose down
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Production Deployment

```bash
# Create .env file
cp .env.example .env
# Edit .env with secure passwords

# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d
```

## Local Development (without Docker)

### 1. Start PostgreSQL with pgvector

```bash
docker run -d --name tree-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=tree_gallery \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Identification
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/identify` | Upload photo, get tree match + health |

### Trees
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/trees/register` | Register new tree |
| GET | `/api/v1/trees` | List all trees |
| GET | `/api/v1/trees/{id}` | Get tree details |
| PUT | `/api/v1/trees/{id}` | Update tree |
| DELETE | `/api/v1/trees/{id}` | Delete tree |

### Photos
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/trees/{id}/photos` | Get tree photos |
| POST | `/api/v1/trees/{id}/photos` | Add photo |
| DELETE | `/api/v1/photos/{id}` | Delete photo |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | API health check |

## Database Schema

### trees
- `id` - Primary key
- `name` - Tree name/label
- `location_lat`, `location_lon` - GPS coordinates
- `species` - Tree species
- `notes` - Additional notes
- `created_at`, `updated_at` - Timestamps

### photos
- `id` - Primary key
- `tree_id` - Foreign key to trees
- `file_path` - Image storage path
- `embedding` - **pgvector Vector(1024)** for similarity search
- `health_assessment` - JSON health data
- `captured_at`, `created_at` - Timestamps

## Vector Similarity Search

Uses pgvector's cosine distance operator:

```sql
SELECT tree_id, 1 - (embedding <=> query_embedding) / 2 as similarity
FROM photos
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

## Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app
│   ├── config.py         # Settings
│   ├── database.py       # PostgreSQL + pgvector
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── routers/          # API endpoints
│   └── services/         # Business logic
├── docker-compose.yml    # Dev deployment
├── docker-compose.prod.yml # Production
├── Dockerfile
├── init.sql              # pgvector extension
├── requirements.txt
└── .env.example
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `UPLOAD_DIR` | `./uploads` | Image storage path |
| `CORS_ORIGINS` | `["*"]` | Allowed origins |
| `EMBEDDING_DIM` | `1024` | ConvNeXt embedding size |
| `SIMILARITY_THRESHOLD` | `0.7` | Match threshold |

## Next Steps

1. Wire up ConvNeXt model in `reid_service.py`
2. Wire up health model in `health_service.py`
3. Add image preprocessing pipeline
4. Add authentication (JWT/API keys)
5. Add rate limiting
6. Set up CI/CD
