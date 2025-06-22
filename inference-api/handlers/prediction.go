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
		meta_id, model_id, predicted_ef,

		volume_mean, volume_var, volume_std, volume_range, volume_ratio,

		length_mean, length_std, length_range, length_ratio,

		area_mean, area_std, area_range, area_ratio,

		magnitude_mean, magnitude_var, magnitude_std, magnitude_range,

		divergence_mean, divergence_var, divergence_std, divergence_range,

		vorticity_mean, vorticity_var, vorticity_std, vorticity_range,

		irrot_energy_mean, irrot_energy_var, irrot_energy_std, irrot_energy_range,

		soleno_energy_mean, soleno_energy_var, soleno_energy_std, soleno_energy_range,

		combined_flow_index_mean, combined_flow_index_var, combined_flow_index_std, combined_flow_index_range,

		dice_mean, dice_var, dice_std, dice_range

	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

	ON CONFLICT(meta_id, model_id) DO UPDATE SET
		predicted_ef = excluded.predicted_ef,

		volume_mean = excluded.volume_mean,
		volume_var = excluded.volume_var,
		volume_std = excluded.volume_std,
		volume_range = excluded.volume_range,
		volume_ratio = excluded.volume_ratio,

		length_mean = excluded.length_mean,
		length_std = excluded.length_std,
		length_range = excluded.length_range,
		length_ratio = excluded.length_ratio,

		area_mean = excluded.area_mean,
		area_std = excluded.area_std,
		area_range = excluded.area_range,
		area_ratio = excluded.area_ratio,

		magnitude_mean = excluded.magnitude_mean,
		magnitude_var = excluded.magnitude_var,
		magnitude_std = excluded.magnitude_std,
		magnitude_range = excluded.magnitude_range,

		divergence_mean = excluded.divergence_mean,
		divergence_var = excluded.divergence_var,
		divergence_std = excluded.divergence_std,
		divergence_range = excluded.divergence_range,

		vorticity_mean = excluded.vorticity_mean,
		vorticity_var = excluded.vorticity_var,
		vorticity_std = excluded.vorticity_std,
		vorticity_range = excluded.vorticity_range,

		irrot_energy_mean = excluded.irrot_energy_mean,
		irrot_energy_var = excluded.irrot_energy_var,
		irrot_energy_std = excluded.irrot_energy_std,
		irrot_energy_range = excluded.irrot_energy_range,

		soleno_energy_mean = excluded.soleno_energy_mean,
		soleno_energy_var = excluded.soleno_energy_var,
		soleno_energy_std = excluded.soleno_energy_std,
		soleno_energy_range = excluded.soleno_energy_range,

		combined_flow_index_mean = excluded.combined_flow_index_mean,
		combined_flow_index_var = excluded.combined_flow_index_var,
		combined_flow_index_std = excluded.combined_flow_index_std,
		combined_flow_index_range = excluded.combined_flow_index_range,

		dice_mean = excluded.dice_mean,
		dice_var = excluded.dice_var,
		dice_std = excluded.dice_std,
		dice_range = excluded.dice_range
	`,
		metaID, modelID, req.PredictedEF,

		req.VolumeMean, req.VolumeVar, req.VolumeStd, req.VolumeRange, req.VolumeRatio,

		req.LengthMean, req.LengthStd, req.LengthRange, req.LengthRatio,

		req.AreaMean, req.AreaStd, req.AreaRange, req.AreaRatio,

		req.MagnitudeMean, req.MagnitudeVar, req.MagnitudeStd, req.MagnitudeRange,

		req.DivergenceMean, req.DivergenceVar, req.DivergenceStd, req.DivergenceRange,

		req.VorticityMean, req.VorticityVar, req.VorticityStd, req.VorticityRange,

		req.IrrotEnergyMean, req.IrrotEnergyVar, req.IrrotEnergyStd, req.IrrotEnergyRange,

		req.SolenoEnergyMean, req.SolenoEnergyVar, req.SolenoEnergyStd, req.SolenoEnergyRange,

		req.CombinedFlowIndexMean, req.CombinedFlowIndexVar, req.CombinedFlowIndexStd, req.CombinedFlowIndexRange,

		req.DiceMean, req.DiceVar, req.DiceStd, req.DiceRange,
	)
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
