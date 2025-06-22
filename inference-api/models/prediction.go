package models

type PredictionRequest struct {
	ModelName string  `json:"model_name"`
	Filename  string  `json:"filename"`
	Split     string  `json:"split"`
	TrueEF    float64 `json:"true_ef"`
	PredictedEF float64 `json:"predicted_ef"`

	VolumeMean  float64 `json:"volume_mean"`
	VolumeVar   float64 `json:"volume_var"`
	VolumeStd   float64 `json:"volume_std"`
	VolumeRange float64 `json:"volume_range"`
	VolumeRatio float64 `json:"volume_ratio"`

	LengthMean  float64 `json:"length_mean"`
	LengthStd   float64 `json:"length_std"`
	LengthRange float64 `json:"length_range"`
	LengthRatio float64 `json:"length_ratio"`

	AreaMean    float64 `json:"area_mean"`
	AreaStd     float64 `json:"area_std"`
	AreaRange   float64 `json:"area_range"`
	AreaRatio   float64 `json:"area_ratio"`

	MagnitudeMean   float64 `json:"magnitude_mean"`
	MagnitudeVar    float64 `json:"magnitude_var"`
	MagnitudeStd    float64 `json:"magnitude_std"`
	MagnitudeRange  float64 `json:"magnitude_range"`

	DivergenceMean  float64 `json:"divergence_mean"`
	DivergenceVar   float64 `json:"divergence_var"`
	DivergenceStd   float64 `json:"divergence_std"`
	DivergenceRange float64 `json:"divergence_range"`

	VorticityMean   float64 `json:"vorticity_mean"`
	VorticityVar    float64 `json:"vorticity_var"`
	VorticityStd    float64 `json:"vorticity_std"`
	VorticityRange  float64 `json:"vorticity_range"`

	IrrotEnergyMean   float64 `json:"irrot_energy_mean"`
	IrrotEnergyVar    float64 `json:"irrot_energy_var"`
	IrrotEnergyStd    float64 `json:"irrot_energy_std"`
	IrrotEnergyRange  float64 `json:"irrot_energy_range"`

	SolenoEnergyMean   float64 `json:"soleno_energy_mean"`
	SolenoEnergyVar    float64 `json:"soleno_energy_var"`
	SolenoEnergyStd    float64 `json:"soleno_energy_std"`
	SolenoEnergyRange  float64 `json:"soleno_energy_range"`

	CombinedFlowIndexMean   float64 `json:"combined_flow_index_mean"`
	CombinedFlowIndexVar    float64 `json:"combined_flow_index_var"`
	CombinedFlowIndexStd    float64 `json:"combined_flow_index_std"`
	CombinedFlowIndexRange  float64 `json:"combined_flow_index_range"`

	DiceMean    float64 `json:"dice_mean"`
	DiceVar     float64 `json:"dice_var"`
	DiceStd     float64 `json:"dice_std"`
	DiceRange   float64 `json:"dice_range"`
}
