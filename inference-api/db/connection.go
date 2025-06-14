package db

import (
	"database/sql"
	"log"

	_ "github.com/mattn/go-sqlite3"
	"inference-api/config"
)

func Connect() *sql.DB {
	db, err := sql.Open("sqlite3", config.DBPath)
	if err != nil {
		log.Fatal("Failed to connect to DB:", err)
	}
	return db
}