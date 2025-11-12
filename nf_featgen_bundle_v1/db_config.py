# Rename this file to db_config.py and fill values.
# Used by featgen.db_utils to build a SQLAlchemy engine.

DB_CONFIG = {
    # Either provide a full URL OR individual fields below.
    # "url": "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME",
    "user":     "postgres",
    "password": "z",
    "host":     "localhost",
    "port":     5432,
    "dbname":   "postgres",
}

# Prefix for table names, e.g., "" or "public." (with schema if desired)
TABLE_PREFIX = "public."


# Linux/Mac は export PGHOST=... でOK
