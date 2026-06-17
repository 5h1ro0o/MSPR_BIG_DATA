"""
Configuration centralisée via Pydantic Settings.
Lit les valeurs depuis .env ou les variables d'environnement.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from security.encryption import DataEncryptor


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("elections_idf", alias="POSTGRES_DB")
    postgres_user: str = Field("etl_admin", alias="POSTGRES_USER")
    postgres_password: str = Field(
        "elections2022", alias="POSTGRES_PASSWORD"
    )  # NOSONAR

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    data_root: Path = Field(Path("/data"), alias="DATA_ROOT")

    @property
    def elections_dir(self) -> Path:
        return self.data_root / "elections"

    @property
    def candidats_file(self) -> Path:
        return self.data_root / "candidats_results.csv"

    @property
    def demographique_file(self) -> Path:
        return self.data_root / "demographique +csp" / "dossier_complet.csv"

    @property
    def chomage_hist_file(self) -> Path:
        return (
            self.data_root
            / "chomage"
            / "chomage-nombre-de-chomeurs-exhaustif-des-communes-dile-de-france-donnee-insee.csv"
        )

    @property
    def emploi_file(self) -> Path:
        return self.data_root / "chomage" / "base-cc-emploi-pop-active-2022.csv"

    @property
    def pauvrete_file(self) -> Path:
        return self.data_root / "pauvreté" / "base-cc-filosofi-2017.csv"

    @property
    def output_file(self) -> Path:
        return self.data_root / "dataset_elections_2022_idf.csv"

    iqr_factor: float = Field(3.0, alias="IQR_FACTOR")
    chunk_size: int = Field(200_000, alias="CHUNK_SIZE")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(Path("logs"), alias="LOG_DIR")

    idf_depts: frozenset[str] = frozenset(
        {"75", "77", "78", "91", "92", "93", "94", "95"}
    )

    # Chiffrement symétrique AES-128 (Fernet)
    encryption_key: str = Field("", alias="ENCRYPTION_KEY")
    encryption_enabled: bool = Field(False, alias="ENCRYPTION_ENABLED")

    @property
    def encryptor(self) -> "DataEncryptor | None":
        """
        Retourne un DataEncryptor si le chiffrement est activé et qu'une clé est présente.
        Retourne None sinon (pipeline continue sans chiffrement).
        """
        if not self.encryption_enabled or not self.encryption_key:
            return None
        from security.encryption import DataEncryptor

        return DataEncryptor(self.encryption_key.encode("utf-8"))


settings = Settings()
