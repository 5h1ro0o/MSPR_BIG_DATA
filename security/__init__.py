"""
security/ — Chiffrement et gestion des clés.

Modules :
  encryption.py  — DataEncryptor (Fernet / AES-128-CBC + HMAC-SHA256)
  key_manager.py — KeyManager (génération, dérivation PBKDF2, rotation)

Utilisation rapide :
    from security.encryption import DataEncryptor
    enc = DataEncryptor.from_env()
    token = enc.encrypt("Paris")
    original = enc.decrypt(token)
"""

from security.encryption import SENSITIVE_DB_COLUMNS, DataEncryptor
from security.key_manager import KeyManager

__all__ = ["DataEncryptor", "KeyManager", "SENSITIVE_DB_COLUMNS"]
