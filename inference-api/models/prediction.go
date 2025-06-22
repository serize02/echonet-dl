package models

type PredictionRequest struct {
	Filename         string  `json:"filename" binding:"required"`
	Split            string  `json:"split" binding:"required"`
	TrueEF           float64 `json:"true_ef" binding:"required"`
	PredictedEF      float64 `json:"predicted_ef" binding:"required"`
	ModelName        string  `json:"model_name" binding:"required"`

	VolumeRange      float64 `json:"volume_range" binding:"required"`
	VolumeMean       float64 `json:"volume_mean" binding:"required"`
	VolumeStd        float64 `json:"volume_std" binding:"required"`
	VolumeMax        float64 `json:"volume_max" binding:"required"`
	VolumeMin        float64 `json:"volume_min" binding:"required"`
	VolumeRatio      float64 `json:"volume_ratio" binding:"required"`

	LengthMean       float64 `json:"length_mean" binding:"required"`
	LengthStd        float64 `json:"length_std" binding:"required"`
	LengthRange      float64 `json:"length_range" binding:"required"`

	AreaMean         float64 `json:"area_mean" binding:"required"`
	AreaStd          float64 `json:"area_std" binding:"required"`
	AreaRange        float64 `json:"area_range" binding:"required"`

	MeanMagnitude    float64 `json:"mean_magnitude" binding:"required"`
	VarMagnitude     float64 `json:"var_magnitude" binding:"required"`
	StdMagnitude     float64 `json:"std_magnitude" binding:"required"`
	MaxMagnitude     float64 `json:"max_magnitude" binding:"required"`

	MeanDivergence   float64 `json:"mean_divergence" binding:"required"`
	VarDivergence    float64 `json:"var_divergence" binding:"required"`
	StdDivergence    float64 `json:"std_divergence" binding:"required"`
	MaxDivergence    float64 `json:"max_divergence" binding:"required"`

	MeanDice         float64 `json:"mean_dice" binding:"required"`
	VarDice          float64 `json:"var_dice" binding:"required"`
	StdDice          float64 `json:"std_dice" binding:"required"`
	MinDice          float64 `json:"min_dice" binding:"required"`
}