package models

type PredictionRequest struct {
	Filename    string  `json:"filename" binding:"required"`
	TrueEF      float64 `json:"true_ef" binding:"required"`
	PredictedEF float64 `json:"predicted_ef" binding:"required"`
	ModelName   string  `json:"model_name" binding:"required"`
}