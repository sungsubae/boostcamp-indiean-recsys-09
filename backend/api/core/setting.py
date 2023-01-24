from yaml import load, FullLoader

from pydantic import BaseSettings, Field
from typing import Optional
from enum import Enum


class ConfigEnv(str, Enum):
    DEV = "dev"
    PROD = "prod"


class SteamConfig(BaseSettings):
    apikey: str = Field(default="00000000000000000000000000000000", env="steam")


class DBConfig(BaseSettings):
    base: str = Field(default="postgresql://", env="db")
    user: str = Field(default=None, env="db")
    password: str = Field(default=None, env="db")
    ip: str = Field(default="localhost", env="db")
    port: str = Field(default="5432", env="db")
    database: str = Field(default="database", env="db")

    @property
    def url(self) -> str:
        return f"{self.base}{self.user}:{self.password}@{self.ip}:{self.port}/{self.database}"


class AppConfig(BaseSettings):
    env: ConfigEnv = Field(default="dev", env="env")
    steam: SteamConfig = SteamConfig()
    db: DBConfig = DBConfig()

with open("config.yaml", "r") as f:
    raw_config = load(f, FullLoader)

config = AppConfig(**raw_config)

assert config.env == "dev"
assert len(config.steam.apikey) == 32