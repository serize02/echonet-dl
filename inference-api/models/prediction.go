package models

type PredictionRequest struct {
	Filename      		string  `json:"filename" binding:"required"`
	Split         		string  `json:"split" binding:"required"`
	TrueEF        		float64 `json:"true_ef" binding:"required"`
	PredictedEF   		float64 `json:"predicted_ef" binding:"required"`
	ModelName     		string  `json:"model_name" binding:"required"`
	VolumeRatio   		float64 `json:"volume_ratio" binding:"required"`
	LengthRatio   		float64 `json:"length_ratio" binding:"required"`
	DiceOverlapStd  	float64 `json:"dice_overlap_std" binding:"required"`
	DiceOverlapRatio   	float64 `json:"dice_overlap_ratio" binding:"required"`
}
