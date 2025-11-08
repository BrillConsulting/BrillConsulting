# Go Microservices Portfolio

High-performance microservices built with Go.

## Projects

### 1. RESTful API
Production-ready REST API with Fiber framework.

**Features:**
- Clean architecture (handler → service → repository)
- GORM for database operations
- Middleware (CORS, rate limiting, logging)
- Graceful error handling
- UUID-based IDs
- Pagination support

**Tech Stack:** Go, Fiber, GORM, PostgreSQL

### 2. gRPC Service (Coming Soon)
High-performance RPC with Protocol Buffers

### 3. Distributed Cache (Coming Soon)
Redis-based caching layer with replication

## Technologies

- **Framework:** Fiber (Express-like for Go)
- **ORM:** GORM
- **Database:** PostgreSQL
- **Tools:** Go modules, air (hot reload)

## Run

```bash
go mod download
go run main.go
```

## API Endpoints

```
GET    /api/v1/users
GET    /api/v1/users/:id
POST   /api/v1/users
PUT    /api/v1/users/:id
DELETE /api/v1/users/:id
```
