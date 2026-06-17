"""Tests unitaires — config/settings.py."""

from pathlib import Path

from config.settings import Settings


class TestSettingsDefaults:
    def test_default_host(self):
        s = Settings()
        assert s.postgres_host == "localhost"

    def test_default_port(self):
        s = Settings()
        assert s.postgres_port == 5432

    def test_default_db(self):
        s = Settings()
        assert s.postgres_db == "elections_idf"

    def test_default_user(self):
        s = Settings()
        assert s.postgres_user == "etl_admin"

    def test_default_iqr_factor(self):
        s = Settings()
        assert s.iqr_factor == 3.0

    def test_default_chunk_size(self):
        s = Settings()
        assert s.chunk_size == 200_000

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_encryption_disabled_by_default(self):
        s = Settings()
        assert s.encryption_enabled is False


class TestSettingsProperties:
    def test_database_url_format(self):
        s = Settings()
        url = s.database_url
        assert url.startswith("postgresql+psycopg2://")
        assert "@" in url
        assert "elections_idf" in url

    def test_database_url_contains_host(self):
        s = Settings()
        assert "localhost" in s.database_url

    def test_elections_dir(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert s.elections_dir == Path("/tmp/data/elections")

    def test_candidats_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert s.candidats_file == Path("/tmp/data/candidats_results.csv")

    def test_output_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert s.output_file == Path("/tmp/data/dataset_elections_2022_idf.csv")

    def test_demographique_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert "dossier_complet.csv" in str(s.demographique_file)

    def test_chomage_hist_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert "chomage" in str(s.chomage_hist_file)

    def test_pauvrete_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert "filosofi" in str(s.pauvrete_file)

    def test_emploi_file(self):
        s = Settings(DATA_ROOT=Path("/tmp/data"))
        assert "emploi" in str(s.emploi_file)


class TestSettingsEncryptor:
    def test_encryptor_none_when_disabled(self):
        s = Settings()
        assert s.encryptor is None

    def test_encryptor_none_when_no_key(self):
        s = Settings(ENCRYPTION_ENABLED=True, ENCRYPTION_KEY="")
        assert s.encryptor is None

    def test_encryptor_returns_instance_when_enabled(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode("utf-8")
        s = Settings(ENCRYPTION_ENABLED=True, ENCRYPTION_KEY=key)
        enc = s.encryptor
        assert enc is not None


class TestSettingsIdfDepts:
    def test_idf_depts_count(self):
        s = Settings()
        assert len(s.idf_depts) == 8

    def test_paris_in_idf(self):
        s = Settings()
        assert "75" in s.idf_depts

    def test_all_expected_depts(self):
        s = Settings()
        expected = {"75", "77", "78", "91", "92", "93", "94", "95"}
        assert s.idf_depts == expected


class TestSettingsEnvOverride:
    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "db_server")
        s = Settings()
        assert s.postgres_host == "db_server"

    def test_port_from_env(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        s = Settings()
        assert s.postgres_port == 5433

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"
