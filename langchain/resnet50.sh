#!/bin/bash

# Usage: ./export_predictions.sh [database_file] [model_name] [output_file]
# Defaults: data/inference.db, ResNet50-UNet, data/inference.csv

DB_FILE=${1:-data/inference.db}
MODEL_NAME=${2:-ResNet50-UNet}
OUTPUT_FILE=${3:-data/inference.csv}

sqlite3 "$DB_FILE" -header -csv "
SELECT 
    meta.filename,
    meta.split,
    meta.true_ef,
    predictions.predicted_ef,

    predictions.volume_range,
    predictions.volume_mean,
    predictions.volume_std,
    predictions.volume_max,
    predictions.volume_min,
    predictions.volume_ratio,

    predictions.length_mean,
    predictions.length_std,
    predictions.length_range,

    predictions.area_mean,
    predictions.area_std,
    predictions.area_range,

    predictions.mean_magnitude,
    predictions.var_magnitude,
    predictions.std_magnitude,
    predictions.max_magnitude,

    predictions.mean_divergence,
    predictions.var_divergence,
    predictions.std_divergence,
    predictions.max_divergence,

    predictions.mean_dice,
    predictions.var_dice,
    predictions.std_dice,
    predictions.min_dice

FROM predictions
JOIN meta ON predictions.meta_id = meta.id
JOIN models ON predictions.model_id = models.id
WHERE models.name = '$MODEL_NAME'
" > "$OUTPUT_FILE"

echo "Exported predictions to $OUTPUT_FILE"
