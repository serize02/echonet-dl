#!/bin/bash

DB_FILE=${1:-dynamic.db}
MODEL_NAME=${2:-ResNet50-UNet}
OUTPUT_FILE=${3:-dynamic.csv}

sqlite3 "$DB_FILE" -header -csv "
SELECT 
    meta.filename,
    meta.split,
    meta.true_ef,
    predictions.predicted_ef,
    predictions.volume_ratio,
    predictions.length_ratio,
    predictions.dice_overlap_std,
    predictions.dice_overlap_ratio,
    predictions.predicted_bias
FROM predictions
JOIN meta ON predictions.meta_id = meta.id
JOIN models ON predictions.model_id = models.id
WHERE models.name = '$MODEL_NAME'
" > "$OUTPUT_FILE"

echo "Exported predictions to $OUTPUT_FILE"
