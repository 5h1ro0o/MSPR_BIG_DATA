"""
Chiffrement symétrique AES-128 (Fernet) pour les données sensibles du pipeline.

Algorithme : Fernet = AES-128-CBC + HMAC-SHA256 + IV aléatoire par message
              (bibliothèque `cryptography` — implémentation validée NIST)

Colonnes chiffrées avant écriture en base :
  - libelle_commune, libelle_departement  (quasi-identifiants géographiques)

Format de stockage : "enc:<token_base64url>"
  → préfixe détectable, longueur variable, authentifié (HMAC)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

from monitoring.logger import get_logger

log = get_logger(__name__)

ENC_PREFIX = "enc:"

# Colonnes considérées sensibles dans gold.ml_features et gold.model_predictions
SENSITIVE_DB_COLUMNS: list[str] = ["libelle_commune", "libelle_departement"]


class DataEncryptor:
    """
    Chiffreur symétrique basé sur Fernet (AES-128-CBC + HMAC-SHA256).

    Chaque valeur chiffrée est authentifiée : toute modification tiers
    est détectée à la déchiffrement (InvalidToken levée).

    Exemple :
        enc = DataEncryptor.from_env()
        chiffre  = enc.encrypt("Paris")          # "enc:gAAAAAB..."
        original = enc.decrypt(chiffre)          # "Paris"
        df_sec   = enc.encrypt_dataframe(df, ["libelle_commune"])
    """

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    # ── Constructeurs ─────────────────────────────────────────────────────────

    @classmethod
    def from_key_bytes(cls, key: bytes) -> "DataEncryptor":
        return cls(key)

    @classmethod
    def from_env(cls, env_var: str = "ENCRYPTION_KEY") -> "DataEncryptor":
        """
        Charge la clé depuis la variable d'environnement ENCRYPTION_KEY.
        Levée EnvironmentError si absente — ne jamais générer silencieusement.
        """
        raw = os.environ.get(env_var, "")
        if not raw:
            raise EnvironmentError(
                f"Variable '{env_var}' manquante dans l'environnement.\n"
                "Générez une clé : python -m security.key_manager generate"
            )
        return cls(raw.encode())

    # ── Chiffrement / déchiffrement unitaire ─────────────────────────────────

    def encrypt(self, value: str) -> str:
        """
        Chiffre une chaîne. Retourne 'enc:<token>'.
        Les valeurs nulles/vides sont retournées intactes.
        """
        if value is None or str(value).strip() in ("", "nan", "None"):
            return value
        token = self._fernet.encrypt(str(value).encode("utf-8"))
        return ENC_PREFIX + token.decode("utf-8")

    def decrypt(self, value: str) -> str:
        """
        Déchiffre une valeur 'enc:<token>'.
        Les valeurs sans préfixe sont retournées intactes (idempotent).
        """
        if not isinstance(value, str) or not value.startswith(ENC_PREFIX):
            return value
        try:
            raw = self._fernet.decrypt(value[len(ENC_PREFIX):].encode("utf-8"))
            return raw.decode("utf-8")
        except InvalidToken:
            log.warning("Déchiffrement impossible — token invalide ou clé incorrecte.")
            return value

    def is_encrypted(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith(ENC_PREFIX)

    # ── Chiffrement / déchiffrement DataFrame ─────────────────────────────────

    def encrypt_dataframe(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """
        Chiffre les colonnes spécifiées. Retourne une copie du DataFrame.
        Les colonnes absentes ou déjà chiffrées sont ignorées.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            already = df[col].apply(
                lambda v: isinstance(v, str) and v.startswith(ENC_PREFIX)
            ).all()
            if already:
                continue
            df[col] = df[col].apply(
                lambda v: self.encrypt(str(v)) if pd.notna(v) else v
            )
            log.info(f"  [enc] Colonne '{col}' chiffrée — {len(df):,} lignes")
        return df

    def decrypt_dataframe(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """Déchiffre les colonnes spécifiées. Retourne une copie."""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda v: self.decrypt(str(v)) if pd.notna(v) else v
            )
        return df

    # ── Chiffrement / déchiffrement fichier ──────────────────────────────────

    def encrypt_file(self, path: Path) -> Path:
        """
        Chiffre un fichier et écrit <path>.enc.
        Le fichier original n'est pas supprimé.

        Returns:
            Chemin du fichier chiffré.
        """
        data = path.read_bytes()
        encrypted = self._fernet.encrypt(data)
        out = path.with_suffix(path.suffix + ".enc")
        out.write_bytes(encrypted)
        log.info(
            f"  [enc] Fichier chiffré : {out.name} "
            f"({len(data):,} → {len(encrypted):,} bytes)"
        )
        return out

    def decrypt_file(self, path: Path) -> bytes:
        """Déchiffre un fichier .enc et retourne les bytes originaux."""
        return self._fernet.decrypt(path.read_bytes())

    def decrypt_file_to(self, encrypted_path: Path, output_path: Path) -> None:
        """Déchiffre un fichier .enc et écrit le résultat dans output_path."""
        output_path.write_bytes(self.decrypt_file(encrypted_path))
        log.info(f"  [enc] Fichier déchiffré : {output_path.name}")
