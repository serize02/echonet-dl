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
	    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
	    meta_id                  INTEGER NOT NULL,
	    model_id                 INTEGER NOT NULL,
	    predicted_ef             REAL NOT NULL,

	    volume_mean              REAL NOT NULL,
	    volume_var               REAL NOT NULL,
	    volume_std               REAL NOT NULL,
	    volume_range             REAL NOT NULL,
	    volume_ratio             REAL NOT NULL,

	    length_mean              REAL NOT NULL,
	    length_std               REAL NOT NULL,
	    length_range             REAL NOT NULL,
	    length_ratio             REAL NOT NULL,

	    area_mean                REAL NOT NULL,
	    area_std                 REAL NOT NULL,
	    area_range               REAL NOT NULL,
	    area_ratio               REAL NOT NULL,

	    magnitude_mean           REAL NOT NULL,
	    magnitude_var            REAL NOT NULL,
	    magnitude_std            REAL NOT NULL,
	    magnitude_range          REAL NOT NULL,

	    divergence_mean          REAL NOT NULL,
	    divergence_var           REAL NOT NULL,
	    divergence_std           REAL NOT NULL,
	    divergence_range         REAL NOT NULL,

	    vorticity_mean           REAL NOT NULL,
	    vorticity_var            REAL NOT NULL,
	    vorticity_std            REAL NOT NULL,
	    vorticity_range          REAL NOT NULL,

	    irrot_energy_mean        REAL NOT NULL,
	    irrot_energy_var         REAL NOT NULL,
	    irrot_energy_std         REAL NOT NULL,
	    irrot_energy_range       REAL NOT NULL,

	    soleno_energy_mean       REAL NOT NULL,
	    soleno_energy_var        REAL NOT NULL,
	    soleno_energy_std        REAL NOT NULL,
	    soleno_energy_range      REAL NOT NULL,

	    combined_flow_index_mean REAL NOT NULL,
	    combined_flow_index_var  REAL NOT NULL,
	    combined_flow_index_std  REAL NOT NULL,
	    combined_flow_index_range REAL NOT NULL,

	    dice_mean                REAL NOT NULL,
	    dice_var                 REAL NOT NULL,
	    dice_std                 REAL NOT NULL,
	    dice_range               REAL NOT NULL,

	    FOREIGN KEY (meta_id) REFERENCES meta(id),
	    FOREIGN KEY (model_id) REFERENCES models(id),
	    UNIQUE (meta_id, model_id)
	);`

	_, err := db.Exec(schema)
	return err
}
