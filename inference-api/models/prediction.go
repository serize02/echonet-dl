package models

type PredictionRequest struct {
	Filename      string  `json:"filename" binding:"required"`
	Split         string  `json:"split" binding:"required"`
	TrueEF        float64 `json:"true_ef" binding:"required"`
	PredictedEF   float64 `json:"predicted_ef" binding:"required"`
	ModelName     string  `json:"model_name" binding:"required"`
	VolumeRatio   float64 `json:"volume_ratio" binding:"required"`
	LengthRatio   float64 `json:"length_ratio" binding:"required"`
	// PredictedBias float64 `json:"predicted_bias" binding:"required"` -> bias-predictor branch
}
