package db

import "database/sql"

func SeedModels(db *sql.DB) error {
	models := []string{"ResNet50-UNet", "UNet-baseline"}
	for _, name := range models {
		_, err := db.Exec(`INSERT OR IGNORE INTO models (name) VALUES (?)`, name)
		if err != nil {
			return err
		}
	}
	return nil
}