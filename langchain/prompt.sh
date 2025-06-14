#!/bin/bash

SCHEMA="Database schema:
Table 'meta': id (INTEGER PRIMARY KEY), filename (TEXT), true_ef (REAL)
Table 'models': id (INTEGER PRIMARY KEY), name (TEXT)
Table 'predictions': id (INTEGER PRIMARY KEY), meta_id (INTEGER), model_id (INTEGER), predicted_ef (REAL)"

QUERY="top 10 files with a significant different between true_ef and predicted_ef"

FULL_PROMPT="$QUERY"

python langchain/sql_agent.py "$FULL_PROMPT"