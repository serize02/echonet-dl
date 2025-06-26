package main

import (
	"log"

	"github.com/gin-gonic/gin"
	"inference-api/db"
	"inference-api/handlers"
)

func main() {
	log.Println("Inference API server starting...")

	database := db.Connect()
	defer database.Close()

	if err := db.Migrate(database); err != nil {
		log.Fatal("Migration failed:", err)
	}
	log.Println("DB migrated.")

	if err := db.SeedModels(database); err != nil {
		log.Fatal("Seeding failed:", err)
	}
	log.Println("Models seeded.")

	r := gin.Default()
	r.POST("/predict", handlers.PostPrediction(database))

	log.Println("Server running at http://localhost:8080")
	r.Run(":8080")
}
