package db

import (
	"database/sql"
)

func Migrate(db *sql.DB) error {
	schema := `
	CREATE TABLE IF NOT EXISTS meta (
	    id        INTEGER PRIMARY KEY AUTOINCREMENT,
	    filename  TEXT NOT NULL UNIQUE,
	    split     TEXT NOT NULL,
	    true_ef   REAL NOT NULL
	);

	CREATE TABLE IF NOT EXISTS models (
	    id    INTEGER PRIMARY KEY AUTOINCREMENT,
	    name  TEXT NOT NULL UNIQUE
	);

	CREATE TABLE IF NOT EXISTS predictions (
	    id                INTEGER PRIMARY KEY AUTOINCREMENT,
	    meta_id           INTEGER NOT NULL,
	    model_id          INTEGER NOT NULL,
	    predicted_ef      REAL NOT NULL,

	    volume_range      REAL NOT NULL,
	    volume_mean       REAL NOT NULL,
	    volume_std        REAL NOT NULL,
	    volume_max        REAL NOT NULL,
	    volume_min        REAL NOT NULL,
	    volume_ratio      REAL NOT NULL,

	    length_mean       REAL NOT NULL,
	    length_std        REAL NOT NULL,
	    length_range      REAL NOT NULL,

	    area_mean         REAL NOT NULL,
	    area_std          REAL NOT NULL,
	    area_range        REAL NOT NULL,

	    mean_magnitude    REAL NOT NULL,
	    var_magnitude     REAL NOT NULL,
	    std_magnitude     REAL NOT NULL,
	    max_magnitude     REAL NOT NULL,

	    mean_divergence   REAL NOT NULL,
	    var_divergence    REAL NOT NULL,
	    std_divergence    REAL NOT NULL,
	    max_divergence    REAL NOT NULL,

	    mean_dice         REAL NOT NULL,
	    var_dice          REAL NOT NULL,
	    std_dice          REAL NOT NULL,
	    min_dice          REAL NOT NULL,

	    FOREIGN KEY (meta_id) REFERENCES meta(id),
	    FOREIGN KEY (model_id) REFERENCES models(id),
	    UNIQUE (meta_id, model_id)
	);`

	_, err := db.Exec(schema)
	return err
}
