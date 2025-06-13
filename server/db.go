package main

import (
    "database/sql"
    "log"

    _ "github.com/mattn/go-sqlite3"
)

const dbPath = "../xai-db/inference.db"

func main() {
    db, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        log.Fatal("Failed to open the database:", err)
    }
    defer db.Close()

    createTable := `
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        true_ef REAL NOT NULL,
        predicted_ef REAL NOT NULL
    );
    `
    _, err = db.Exec(createTable)
    if err != nil {
        log.Fatal("Failed to create table:", err)
    }

    log.Println("Database and table created successfully.")
}
