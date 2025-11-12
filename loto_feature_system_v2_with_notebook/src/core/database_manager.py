from __future__ import annotations
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import pandas as pd

@dataclass
class DBConfig:
    username: str
    password: str
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    sslmode: str = "prefer"

    def url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"

def fetch_dataframe(sql: str, cfg: DBConfig) -> pd.DataFrame:
    engine = create_engine(cfg.url())
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)
