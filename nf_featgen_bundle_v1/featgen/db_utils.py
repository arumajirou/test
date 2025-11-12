from __future__ import annotations
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from typing import Dict
from db_config import DB_CONFIG  # user-provided module
import os

def make_url(cfg: Dict) -> str:
    if "url" in cfg:
        return cfg["url"]
    user = cfg.get("user") or cfg.get("username")
    pwd  = cfg.get("password") or cfg.get("pwd") or cfg.get("pass")
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 5432))
    db   = cfg.get("dbname") or cfg.get("database") or cfg.get("db")
    if not (user and pwd and db):
        raise ValueError(f"DB設定を確認してください。渡されたキー: {list(cfg.keys())}")
    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}"

def get_engine():
    url = make_url(DB_CONFIG)
    return create_engine(url, pool_pre_ping=True, future=True)
