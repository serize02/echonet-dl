package main

import (
    "database/sql"
    "encoding/json"
    "log"
    "net/http"

    _ "github.com/mattn/go-sqlite3"
)

const dbPath = "../xai-db/inference.db"

type Prediction struct {
    Filename    string  `json:"filename"`
    TrueEF      float64 `json:"true_ef"`
    PredictedEF float64 `json:"predicted_ef"`
}

func insertPrediction(p Prediction) error {
    db, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        return err
    }
    defer db.Close()

    _, err = db.Exec(`
        INSERT INTO predictions (filename, true_ef, predicted_ef)
        VALUES (?, ?, ?)
    `, p.Filename, p.TrueEF, p.PredictedEF)
    return err
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var p Prediction
    err := json.NewDecoder(r.Body).Decode(&p)
    if err != nil {
        http.Error(w, "Invalid JSON body", http.StatusBadRequest)
        return
    }

    err = insertPrediction(p)
    if err != nil {
        http.Error(w, "Failed to insert prediction: "+err.Error(), http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusOK)
    w.Write([]byte("Prediction stored successfully."))
}

func main() {
    http.HandleFunc("/predict", predictHandler)
    log.Println("Server is running at http://localhost:8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}