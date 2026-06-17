"""Tests unitaires — monitoring/logger.py et monitoring/metrics.py."""

import json

from monitoring.metrics import PipelineMetrics, StepMetric


class TestStepMetric:
    def test_duration_without_end(self):
        m = StepMetric(step="test")
        assert m.duration_s == 0.0

    def test_duration_with_end(self):
        m = StepMetric(step="test", start_ts=1000.0, end_ts=1005.0)
        assert m.duration_s == 5.0

    def test_finish_sets_end_ts(self):
        m = StepMetric(step="test")
        assert m.end_ts is None
        m.finish()
        assert m.end_ts is not None
        assert m.duration_s > 0.0

    def test_default_fields(self):
        m = StepMetric(step="extract")
        assert m.rows_in == 0
        assert m.rows_out == 0
        assert m.cols_out == 0
        assert m.nulls_before == 0
        assert m.nulls_after == 0
        assert m.error is None

    def test_step_name_stored(self):
        m = StepMetric(step="transform")
        assert m.step == "transform"


class TestPipelineMetrics:
    def test_start_step_adds_metric(self):
        pm = PipelineMetrics(run_id="test_run")
        m = pm.start_step("extract")
        assert len(pm.steps) == 1
        assert pm.steps[0].step == "extract"
        assert isinstance(m, StepMetric)

    def test_start_multiple_steps(self):
        pm = PipelineMetrics(run_id="run")
        pm.start_step("extract")
        pm.start_step("transform")
        pm.start_step("load")
        assert len(pm.steps) == 3

    def test_to_dict_structure(self):
        pm = PipelineMetrics(run_id="test_run")
        m = pm.start_step("extract")
        m.rows_out = 100
        m.finish()
        d = pm.to_dict()
        assert d["run_id"] == "test_run"
        assert "started_at" in d
        assert "steps" in d
        assert "total_duration_s" in d
        assert len(d["steps"]) == 1
        assert d["steps"][0]["rows_out"] == 100

    def test_total_duration_sums_steps(self):
        pm = PipelineMetrics(run_id="run")
        m1 = StepMetric(step="s1", start_ts=0.0, end_ts=2.0)
        m2 = StepMetric(step="s2", start_ts=2.0, end_ts=5.0)
        pm.steps = [m1, m2]
        assert pm.to_dict()["total_duration_s"] == 5.0

    def test_empty_steps_total_zero(self):
        pm = PipelineMetrics(run_id="empty")
        assert pm.to_dict()["total_duration_s"] == 0.0

    def test_save_writes_json(self, tmp_path):
        pm = PipelineMetrics(run_id="save_test")
        pm.save(tmp_path)
        files = list(tmp_path.glob("metrics_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["run_id"] == "save_test"

    def test_save_creates_nested_dir(self, tmp_path):
        pm = PipelineMetrics(run_id="run")
        new_dir = tmp_path / "nested" / "metrics"
        pm.save(new_dir)
        assert new_dir.exists()

    def test_run_id_stored(self):
        pm = PipelineMetrics(run_id="abc123")
        assert pm.run_id == "abc123"

    def test_started_at_is_string(self):
        pm = PipelineMetrics(run_id="r")
        assert isinstance(pm.started_at, str)
        assert "T" in pm.started_at  # ISO format contains T


class TestLogger:
    def test_get_logger_returns_bound_logger(self):
        from monitoring.logger import get_logger

        log = get_logger("test_module")
        assert log is not None

    def test_setup_logger_creates_log_dir(self, tmp_path):
        from monitoring.logger import setup_logger

        log_dir = tmp_path / "custom_logs"
        setup_logger(log_level="WARNING", log_dir=log_dir)
        assert log_dir.exists()

    def test_get_logger_does_not_raise(self):
        from monitoring.logger import get_logger

        log = get_logger("my.module")
        log.debug("test debug message")
        log.info("test info message")

    def test_setup_logger_warning_level(self, tmp_path):
        from monitoring.logger import setup_logger

        log_dir = tmp_path / "logs"
        setup_logger(log_level="WARNING", log_dir=log_dir)
        assert log_dir.exists()

    def test_setup_logger_info_level(self, tmp_path):
        from monitoring.logger import setup_logger

        log_dir = tmp_path / "logs_info"
        setup_logger(log_level="INFO", log_dir=log_dir)
        assert log_dir.exists()
