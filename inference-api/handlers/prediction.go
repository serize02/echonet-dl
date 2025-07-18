package handlers

import (
	"database/sql"
	"net/http"

	"github.com/gin-gonic/gin"
	"inference-api/models"
)

func getOrInsertMeta(db *sql.DB, filename string, split string, trueEF float64) (int, error) {
	var id int
	err := db.QueryRow(`SELECT id FROM meta WHERE filename = ?`, filename).Scan(&id)
	if err == sql.ErrNoRows {
		res, err := db.Exec(`INSERT INTO meta (filename, split, true_ef) VALUES (?, ?, ?)`, filename, split, trueEF)
		if err != nil {
			return 0, err
		}
		newID, err := res.LastInsertId()
		return int(newID), err
	}
	return id, err
}

func getModelID(db *sql.DB, modelName string) (int, error) {
	var id int
	err := db.QueryRow(`SELECT id FROM models WHERE name = ?`, modelName).Scan(&id)
	return id, err
}

func insertPrediction(db *sql.DB, metaID, modelID int, req models.PredictionRequest) error {
	_, err := db.Exec(`
	INSERT INTO predictions (
		meta_id, model_id, predicted_ef, volume_ratio, length_ratio, dice_overlap_std, dice_overlap_ratio
	) VALUES (?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(meta_id, model_id) DO UPDATE SET
		predicted_ef     	= excluded.predicted_ef,
		volume_ratio     	= excluded.volume_ratio,
		length_ratio     	= excluded.length_ratio,
		dice_overlap_std  	= excluded.dice_overlap_std,
		dice_overlap_ratio  = excluded.dice_overlap_ratio
	`, metaID, modelID, req.PredictedEF, req.VolumeRatio, req.LengthRatio, req.DiceOverlapStd, req.DiceOverlapRatio)

	return err
}

func PostPrediction(db *sql.DB) gin.HandlerFunc {

	return func(c *gin.Context) {
		var req models.PredictionRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON", "details": err.Error()})
			return
		}

		metaID, err := getOrInsertMeta(db, req.Filename, req.Split, req.TrueEF)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process meta", "details": err.Error()})
			return
		}

		modelID, err := getModelID(db, req.ModelName)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model not found", "model_name": req.ModelName})
			return
		}

		err = insertPrediction(db, metaID, modelID, req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to insert prediction", "details": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"message": "Prediction stored successfully"})
	}
}
