#!/bin/bash

# Usage: ./export_predictions.sh [database_file] [model_name] [output_file]
# Defaults: data/helmholtz_hodge_inference.db, ResNet50-UNet, data/inference_helmholtz.csv

DB_FILE=${1:-data/hh_inference.db}
MODEL_NAME=${2:-ResNet50-UNet}
OUTPUT_FILE=${3:-data/hh_inference.csv}

sqlite3 "$DB_FILE" -header -csv "
SELECT 
    meta.filename,
    meta.split,
    meta.true_ef,
    predictions.predicted_ef,

    predictions.volume_mean,
    predictions.volume_var,
    predictions.volume_std,
    predictions.volume_range,
    predictions.volume_ratio,

    predictions.length_mean,
    predictions.length_std,
    predictions.length_range,
    predictions.length_ratio,

    predictions.area_mean,
    predictions.area_std,
    predictions.area_range,
    predictions.area_ratio,

    predictions.magnitude_mean,
    predictions.magnitude_var,
    predictions.magnitude_std,
    predictions.magnitude_range,

    predictions.divergence_mean,
    predictions.divergence_var,
    predictions.divergence_std,
    predictions.divergence_range,

    predictions.vorticity_mean,
    predictions.vorticity_var,
    predictions.vorticity_std,
    predictions.vorticity_range,

    predictions.irrot_energy_mean,
    predictions.irrot_energy_var,
    predictions.irrot_energy_std,
    predictions.irrot_energy_range,

    predictions.soleno_energy_mean,
    predictions.soleno_energy_var,
    predictions.soleno_energy_std,
    predictions.soleno_energy_range,

    predictions.combined_flow_index_mean,
    predictions.combined_flow_index_var,
    predictions.combined_flow_index_std,
    predictions.combined_flow_index_range,

    predictions.dice_mean,
    predictions.dice_var,
    predictions.dice_std,
    predictions.dice_range

FROM predictions
JOIN meta ON predictions.meta_id = meta.id
JOIN models ON predictions.model_id = models.id
WHERE models.name = '$MODEL_NAME'
" > "$OUTPUT_FILE"

echo "Exported predictions to $OUTPUT_FILE"
