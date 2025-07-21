package db

import "database/sql"

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
	    id              		INTEGER PRIMARY KEY AUTOINCREMENT,
	    meta_id         		INTEGER NOT NULL,
	    model_id        		INTEGER NOT NULL,
	    predicted_ef    		REAL NOT NULL,
	    volume_ratio    		REAL NOT NULL,
	    length_ratio    		REAL NOT NULL,
		dice_overlap_std 		REAL NOT NULL,
		dice_overlap_ratio 		REAL NOT NULL,
		predicted_bias			REAL NOT NULL,
	    FOREIGN KEY (meta_id) 	REFERENCES meta(id),
	    FOREIGN KEY (model_id) 	REFERENCES models(id),
	    UNIQUE (meta_id, model_id)
	);`

	_, err := db.Exec(schema)
	return err
}