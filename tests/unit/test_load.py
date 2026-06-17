"""Tests unitaires — etl/load/csv_writer.py et etl/load/db_loader.py."""

from unittest.mock import MagicMock, patch

import pandas as pd

from etl.load.csv_writer import write_gold_csv


class TestWriteGoldCsv:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = tmp_path / "output.csv"
        write_gold_csv(df, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        out = tmp_path / "sub" / "dir" / "gold.csv"
        write_gold_csv(df, out)
        assert out.exists()

    def test_written_content_correct(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        out = tmp_path / "gold.csv"
        write_gold_csv(df, out)
        loaded = pd.read_csv(out)
        assert list(loaded["x"]) == [10, 20]
        assert list(loaded["y"]) == [30, 40]

    def test_archives_existing_file(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "output.csv"
        out.write_text("old content", encoding="utf-8")
        write_gold_csv(df, out)
        assert out.exists()
        all_files = [f for f in tmp_path.iterdir()]
        assert len(all_files) == 2

    def test_archive_has_timestamp_in_name(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "output.csv"
        out.write_text("old", encoding="utf-8")
        write_gold_csv(df, out)
        archive = [f for f in tmp_path.iterdir() if f != out][0]
        assert "output_" in archive.name

    def test_no_archive_if_no_existing_file(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "fresh.csv"
        write_gold_csv(df, out)
        assert len(list(tmp_path.iterdir())) == 1

    def test_writes_utf8_bom(self, tmp_path):
        df = pd.DataFrame({"commune": ["Île-de-France"]})
        out = tmp_path / "gold.csv"
        write_gold_csv(df, out)
        raw = out.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"

    def test_large_dataframe(self, tmp_path):
        df = pd.DataFrame({"v": range(10_000)})
        out = tmp_path / "big.csv"
        write_gold_csv(df, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 10_000


class TestCheckDbConnection:
    def test_returns_false_on_invalid_url(self):
        from etl.load.db_loader import check_db_connection

        result = check_db_connection("postgresql+psycopg2://bad:bad@localhost:1/bad")
        assert result is False

    def test_returns_true_on_mock_connection(self):
        from etl.load.db_loader import check_db_connection

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("etl.load.db_loader.create_engine", return_value=mock_engine):
            result = check_db_connection("postgresql+psycopg2://u:p@host/db")
        assert result is True

    def test_returns_false_on_exception(self):
        from etl.load.db_loader import check_db_connection

        with patch(
            "etl.load.db_loader.create_engine", side_effect=Exception("conn error")
        ):
            result = check_db_connection("postgresql+psycopg2://u:p@host/db")
        assert result is False
