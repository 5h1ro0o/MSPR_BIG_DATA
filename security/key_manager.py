"""
Gestion du cycle de vie des clés de chiffrement.

Fournit :
  - Génération de clés Fernet (32 bytes aléatoires sécurisés, encodés base64url)
  - Dérivation de clé depuis un mot de passe (PBKDF2-HMAC-SHA256, 480 000 itérations)
  - Écriture dans config/.env
  - Rotation de clé sur un DataFrame existant

Usage CLI :
    python -m security.key_manager generate
    python -m security.key_manager rotate --old-key <ancien> --new-key <nouveau>
"""

from __future__ import annotations

import base64
import re
import secrets
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import pandas as pd

# 480 000 itérations — recommandation OWASP 2024 pour PBKDF2-HMAC-SHA256
_PBKDF2_ITERATIONS = 480_000


class KeyManager:
    """
    Utilitaire de gestion des clés de chiffrement Fernet.

    La clé Fernet est un tableau de 32 bytes aléatoires encodé en base64url
    (44 caractères). Elle est symétrique : une seule clé pour chiffrer ET
    déchiffrer — à conserver hors du dépôt (variable d'environnement ou secret).
    """

    # ── Génération ────────────────────────────────────────────────────────────

    @staticmethod
    def generate_key() -> bytes:
        """
        Génère une clé Fernet aléatoire (32 bytes via os.urandom).

        Returns:
            bytes — clé encodée base64url (44 caractères ASCII)
        """
        return Fernet.generate_key()

    @staticmethod
    def generate_and_print() -> str:
        """Génère une clé, l'affiche sur stdout et la retourne."""
        key = KeyManager.generate_key().decode("utf-8")
        print(f"ENCRYPTION_KEY={key}")
        print("→ Copiez cette ligne dans config/.env et NE la commitez JAMAIS.")
        return key

    # ── Dérivation mot de passe → clé ─────────────────────────────────────────

    @staticmethod
    def derive_key(
        password: str | bytes,
        salt: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """
        Dérive une clé Fernet depuis un mot de passe (PBKDF2-HMAC-SHA256).

        Args:
            password: mot de passe en clair (str ou bytes)
            salt:     sel aléatoire 16 bytes ; si None, en génère un nouveau

        Returns:
            (key_b64url, salt) — la clé prête pour Fernet et le sel utilisé
        """
        if isinstance(password, str):
            password = password.encode("utf-8")
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        raw_key = kdf.derive(password)
        key_b64 = base64.urlsafe_b64encode(raw_key)
        return key_b64, salt

    # ── Persistance .env ──────────────────────────────────────────────────────

    @staticmethod
    def write_key_to_env(
        env_path: Path | str | None = None,
        key: bytes | str | None = None,
    ) -> str:
        """
        Écrit (ou met à jour) ENCRYPTION_KEY dans config/.env.

        Si key est None, génère une nouvelle clé.
        Si la variable existe déjà, la remplace ; sinon l'ajoute en fin de fichier.

        Returns:
            La clé écrite (str, encodage base64url).
        """
        if env_path is None:
            env_path = Path(__file__).parent.parent / "config" / ".env"
        env_path = Path(env_path)

        if key is None:
            key = KeyManager.generate_key()
        if isinstance(key, bytes):
            key = key.decode("utf-8")

        text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        pattern = re.compile(r"^ENCRYPTION_KEY=.*$", re.MULTILINE)

        if pattern.search(text):
            text = pattern.sub(f"ENCRYPTION_KEY={key}", text)
        else:
            text = text.rstrip("\n") + f"\nENCRYPTION_KEY={key}\n"

        env_path.write_text(text, encoding="utf-8")
        print(f"Clé écrite dans {env_path}")
        return key

    # ── Rotation de clé ───────────────────────────────────────────────────────

    @staticmethod
    def rotate_key(
        old_encryptor: "DataEncryptor",  # noqa: F821
        new_key: bytes | str,
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Rotation de clé : déchiffre avec l'ancienne clé, rechiffre avec la nouvelle.

        Args:
            old_encryptor: DataEncryptor initialisé avec l'ancienne clé
            new_key:       nouvelle clé Fernet (bytes ou str base64url)
            df:            DataFrame contenant des valeurs chiffrées
            columns:       colonnes à faire tourner

        Returns:
            DataFrame avec les valeurs rechiffrées avec la nouvelle clé.
        """
        from security.encryption import DataEncryptor

        if isinstance(new_key, str):
            new_key = new_key.encode("utf-8")
        new_enc = DataEncryptor.from_key_bytes(new_key)

        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda v: (
                    new_enc.encrypt(old_encryptor.decrypt(str(v))) if pd.notna(v) else v
                )
            )
            print(f"  [rotate] Colonne '{col}' — {len(df):,} valeurs rechiffrées")
        return df


# ── Entrée CLI ────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m security.key_manager",
        description="Gestion des clés de chiffrement Fernet",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("generate", help="Génère et affiche une nouvelle clé")

    write_p = sub.add_parser("write-env", help="Écrit la clé dans config/.env")
    write_p.add_argument("--key", default=None, help="Clé à écrire (génère si absent)")
    write_p.add_argument("--env", default=None, help="Chemin du fichier .env")

    args = parser.parse_args()

    if args.cmd == "generate":
        KeyManager.generate_and_print()
    elif args.cmd == "write-env":
        KeyManager.write_key_to_env(
            env_path=args.env,
            key=args.key,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
