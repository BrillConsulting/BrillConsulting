/**
 * Go RESTful API with Fiber Framework
 * High-performance microservice with middleware, validation, and database integration
 */

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/google/uuid"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// ===== Models =====
type User struct {
	ID        uuid.UUID `gorm:"type:uuid;primary_key;default:uuid_generate_v4()" json:"id"`
	Name      string    `gorm:"size:100;not null" json:"name" validate:"required,min=2,max=100"`
	Email     string    `gorm:"size:100;unique;not null" json:"email" validate:"required,email"`
	Password  string    `gorm:"size:255;not null" json:"-" validate:"required,min=8"`
	Active    bool      `gorm:"default:true" json:"active"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type CreateUserRequest struct {
	Name     string `json:"name" validate:"required,min=2,max=100"`
	Email    string `json:"email" validate:"required,email"`
	Password string `json:"password" validate:"required,min=8"`
}

type UpdateUserRequest struct {
	Name   string `json:"name" validate:"min=2,max=100"`
	Email  string `json:"email" validate:"email"`
	Active *bool  `json:"active"`
}

type Response struct {
	Success bool        `json:"success"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// ===== Database =====
type Database struct {
	DB *gorm.DB
}

func NewDatabase() (*Database, error) {
	dsn := fmt.Sprintf(
		"host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		getEnv("DB_HOST", "localhost"),
		getEnv("DB_USER", "postgres"),
		getEnv("DB_PASSWORD", "postgres"),
		getEnv("DB_NAME", "users_db"),
		getEnv("DB_PORT", "5432"),
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Auto migrate
	if err := db.AutoMigrate(&User{}); err != nil {
		return nil, fmt.Errorf("failed to migrate database: %w", err)
	}

	return &Database{DB: db}, nil
}

// ===== Repository =====
type UserRepository struct {
	db *gorm.DB
}

func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{db: db}
}

func (r *UserRepository) FindAll(ctx context.Context, limit, offset int) ([]User, error) {
	var users []User
	result := r.db.WithContext(ctx).
		Limit(limit).
		Offset(offset).
		Find(&users)

	return users, result.Error
}

func (r *UserRepository) FindByID(ctx context.Context, id uuid.UUID) (*User, error) {
	var user User
	result := r.db.WithContext(ctx).First(&user, "id = ?", id)

	if result.Error != nil {
		return nil, result.Error
	}

	return &user, nil
}

func (r *UserRepository) FindByEmail(ctx context.Context, email string) (*User, error) {
	var user User
	result := r.db.WithContext(ctx).First(&user, "email = ?", email)

	if result.Error != nil {
		return nil, result.Error
	}

	return &user, nil
}

func (r *UserRepository) Create(ctx context.Context, user *User) error {
	return r.db.WithContext(ctx).Create(user).Error
}

func (r *UserRepository) Update(ctx context.Context, user *User) error {
	return r.db.WithContext(ctx).Save(user).Error
}

func (r *UserRepository) Delete(ctx context.Context, id uuid.UUID) error {
	return r.db.WithContext(ctx).Delete(&User{}, "id = ?", id).Error
}

func (r *UserRepository) Count(ctx context.Context) (int64, error) {
	var count int64
	result := r.db.WithContext(ctx).Model(&User{}).Count(&count)
	return count, result.Error
}

// ===== Service =====
type UserService struct {
	repo *UserRepository
}

func NewUserService(repo *UserRepository) *UserService {
	return &UserService{repo: repo}
}

func (s *UserService) GetAllUsers(ctx context.Context, page, pageSize int) ([]User, error) {
	offset := (page - 1) * pageSize
	return s.repo.FindAll(ctx, pageSize, offset)
}

func (s *UserService) GetUserByID(ctx context.Context, id uuid.UUID) (*User, error) {
	return s.repo.FindByID(ctx, id)
}

func (s *UserService) CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error) {
	// Check if user already exists
	existing, err := s.repo.FindByEmail(ctx, req.Email)
	if err == nil && existing != nil {
		return nil, fmt.Errorf("user with email %s already exists", req.Email)
	}

	// Hash password (in production, use bcrypt)
	hashedPassword := hashPassword(req.Password)

	user := &User{
		ID:        uuid.New(),
		Name:      req.Name,
		Email:     req.Email,
		Password:  hashedPassword,
		Active:    true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.repo.Create(ctx, user); err != nil {
		return nil, err
	}

	return user, nil
}

func (s *UserService) UpdateUser(ctx context.Context, id uuid.UUID, req *UpdateUserRequest) (*User, error) {
	user, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("user not found")
	}

	if req.Name != "" {
		user.Name = req.Name
	}

	if req.Email != "" {
		user.Email = req.Email
	}

	if req.Active != nil {
		user.Active = *req.Active
	}

	user.UpdatedAt = time.Now()

	if err := s.repo.Update(ctx, user); err != nil {
		return nil, err
	}

	return user, nil
}

func (s *UserService) DeleteUser(ctx context.Context, id uuid.UUID) error {
	return s.repo.Delete(ctx, id)
}

// ===== Handlers =====
type UserHandler struct {
	service *UserService
}

func NewUserHandler(service *UserService) *UserHandler {
	return &UserHandler{service: service}
}

func (h *UserHandler) GetUsers(c *fiber.Ctx) error {
	page := c.QueryInt("page", 1)
	pageSize := c.QueryInt("page_size", 10)

	users, err := h.service.GetAllUsers(c.Context(), page, pageSize)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(Response{
			Success: false,
			Error:   err.Error(),
		})
	}

	return c.JSON(Response{
		Success: true,
		Data:    users,
	})
}

func (h *UserHandler) GetUser(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   "invalid user ID",
		})
	}

	user, err := h.service.GetUserByID(c.Context(), id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(Response{
			Success: false,
			Error:   "user not found",
		})
	}

	return c.JSON(Response{
		Success: true,
		Data:    user,
	})
}

func (h *UserHandler) CreateUser(c *fiber.Ctx) error {
	var req CreateUserRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   "invalid request body",
		})
	}

	user, err := h.service.CreateUser(c.Context(), &req)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   err.Error(),
		})
	}

	return c.Status(fiber.StatusCreated).JSON(Response{
		Success: true,
		Message: "user created successfully",
		Data:    user,
	})
}

func (h *UserHandler) UpdateUser(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   "invalid user ID",
		})
	}

	var req UpdateUserRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   "invalid request body",
		})
	}

	user, err := h.service.UpdateUser(c.Context(), id, &req)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(Response{
			Success: false,
			Error:   err.Error(),
		})
	}

	return c.JSON(Response{
		Success: true,
		Message: "user updated successfully",
		Data:    user,
	})
}

func (h *UserHandler) DeleteUser(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(Response{
			Success: false,
			Error:   "invalid user ID",
		})
	}

	if err := h.service.DeleteUser(c.Context(), id); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(Response{
			Success: false,
			Error:   err.Error(),
		})
	}

	return c.Status(fiber.StatusNoContent).Send(nil)
}

// ===== Routes =====
func setupRoutes(app *fiber.App, handler *UserHandler) {
	api := app.Group("/api/v1")

	users := api.Group("/users")
	users.Get("/", handler.GetUsers)
	users.Get("/:id", handler.GetUser)
	users.Post("/", handler.CreateUser)
	users.Put("/:id", handler.UpdateUser)
	users.Delete("/:id", handler.DeleteUser)
}

// ===== Utilities =====
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func hashPassword(password string) string {
	// In production, use bcrypt
	return password // Simplified for example
}

// ===== Main =====
func main() {
	// Initialize database
	database, err := NewDatabase()
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Initialize layers
	userRepo := NewUserRepository(database.DB)
	userService := NewUserService(userRepo)
	userHandler := NewUserHandler(userService)

	// Create Fiber app
	app := fiber.New(fiber.Config{
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}

			return c.Status(code).JSON(Response{
				Success: false,
				Error:   err.Error(),
			})
		},
	})

	// Middleware
	app.Use(recover.New())
	app.Use(logger.New())
	app.Use(cors.New())
	app.Use(limiter.New(limiter.Config{
		Max:        100,
		Expiration: 1 * time.Minute,
	}))

	// Routes
	setupRoutes(app, userHandler)

	// Health check
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status": "ok",
			"time":   time.Now(),
		})
	})

	// Start server
	port := getEnv("PORT", "3000")
	log.Printf("Server starting on port %s", port)
	log.Fatal(app.Listen(":" + port))
}
