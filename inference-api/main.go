package main

import (
	"log"

	"github.com/gin-gonic/gin"
	"inference-api/db"
	"inference-api/handlers"
)

func main() {
	database := db.Connect()
	defer database.Close()

	if err := db.Migrate(database); err != nil {
		log.Fatal("Failed to migrate schema:", err)
	}

	if err := db.SeedModels(database); err != nil {
		log.Fatal("Failed to seed models:", err)
	}

	r := gin.Default()
	r.POST("/predict", handlers.PostPrediction(database))

	log.Println("Server running at http://localhost:8080")
	r.Run(":8080")
}
