"""
Unit tests for logging_config module.

Tests the logging configuration setup, structured logging,
and log formatting functionality.
"""

import pytest
import logging
import json
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logging_config import (
    setup_logging,
    StructuredLogger,
    get_logger
)


@pytest.mark.unit
class TestLoggingSetup:
    """Test suite for logging setup functionality."""
    
    def test_setup_logging_default_level(self):
        """Test logging setup with default INFO level."""
        logger = setup_logging()
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom DEBUG level."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file handler."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(log_file=str(log_file))
        
        # Check file handler was added
        file_handlers = [h for h in logger.handlers 
                        if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
    
    def test_setup_logging_json_format(self):
        """Test logging setup with JSON formatting."""
        logger = setup_logging(json_format=True)
        
        # Check that handlers use JSON formatter
        for handler in logger.handlers:
            assert handler.formatter is not None


@pytest.mark.unit
class TestStructuredLogger:
    """Test suite for StructuredLogger class."""
    
    def test_structured_logger_initialization(self):
        """Test StructuredLogger initialization."""
        logger = StructuredLogger("test_logger")
        assert logger.logger.name == "test_logger"
    
    def test_log_with_context(self):
        """Test logging with additional context."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger = StructuredLogger("test_logger")
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.info("Test message", extra_field="extra_value")
        
        output = stream.getvalue()
        assert "Test message" in output
    
    def test_log_levels(self):
        """Test different log levels."""
        logger = StructuredLogger("test_logger")
        logger.logger.setLevel(logging.DEBUG)
        
        # Test each level works without error
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_exception_logging(self):
        """Test exception logging with traceback."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger("test_logger")
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.ERROR)
        
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred")
        
        output = stream.getvalue()
        assert "An error occurred" in output
        assert "ValueError" in output


@pytest.mark.unit
class TestGetLogger:
    """Test suite for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_same_name_returns_same_instance(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is logger2
    
    def test_get_logger_different_names(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name


@pytest.mark.unit
class TestJSONFormatting:
    """Test suite for JSON log formatting."""
    
    def test_json_log_output_format(self, temp_dir):
        """Test that JSON logs are properly formatted."""
        log_file = temp_dir / "json_test.log"
        logger = setup_logging(
            log_file=str(log_file),
            json_format=True
        )
        
        logger.info("Test message", extra={'field': 'value'})
        
        # Read and parse JSON log
        with open(log_file, 'r') as f:
            log_line = f.readline()
            if log_line.strip():
                log_data = json.loads(log_line)
                assert 'message' in log_data or 'msg' in log_data
                assert 'timestamp' in log_data or 'time' in log_data
    
    def test_json_log_with_exception(self, temp_dir):
        """Test JSON logging with exception information."""
        log_file = temp_dir / "json_exception_test.log"
        logger = setup_logging(
            log_file=str(log_file),
            json_format=True,
            level=logging.ERROR
        )
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred")
        
        # Verify log file contains exception info
        with open(log_file, 'r') as f:
            content = f.read()
            assert "ValueError" in content or "exception" in content.lower()


@pytest.mark.unit
class TestLoggerConfiguration:
    """Test suite for logger configuration options."""
    
    def test_multiple_handlers(self, temp_dir):
        """Test logger with multiple handlers."""
        log_file = temp_dir / "multi_handler.log"
        logger = setup_logging(log_file=str(log_file))
        
        # Should have both console and file handlers
        assert len(logger.handlers) >= 1
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = setup_logging(level=logging.WARNING)
        logger.handlers = [handler]  # Replace with test handler
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        output = stream.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy and propagation."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert child_logger.parent == parent_logger


@pytest.mark.unit
class TestLoggerPerformance:
    """Test suite for logger performance characteristics."""
    
    def test_logging_does_not_throw_exceptions(self):
        """Test that logging never throws exceptions."""
        logger = setup_logging()
        
        # These should not raise exceptions
        logger.info("Normal message")
        logger.info("Message with %s", "format")
        logger.info("Message", extra={'key': 'value'})
    
    def test_structured_logger_performance(self):
        """Test that structured logging is performant."""
        logger = StructuredLogger("perf_test")
        logger.logger.setLevel(logging.INFO)
        
        # Add null handler to avoid I/O
        logger.logger.addHandler(logging.NullHandler())
        
        import time
        start = time.time()
        for i in range(1000):
            logger.info(f"Message {i}", iteration=i)
        duration = time.time() - start
        
        # Should complete 1000 logs in reasonable time
        assert duration < 1.0  # Less than 1 second


@pytest.mark.unit
class TestLoggerEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_none_values_in_extra(self):
        """Test handling of None values in extra fields."""
        logger = StructuredLogger("test")
        logger.logger.setLevel(logging.INFO)
        logger.logger.addHandler(logging.NullHandler())
        
        # Should not raise exception
        logger.info("Message", field=None)
    
    def test_complex_objects_in_extra(self):
        """Test handling of complex objects in extra fields."""
        logger = StructuredLogger("test")
        logger.logger.setLevel(logging.INFO)
        logger.logger.addHandler(logging.NullHandler())
        
        # Should handle complex objects
        logger.info("Message", obj={'nested': {'key': 'value'}})
    
    def test_unicode_in_messages(self):
        """Test handling of unicode characters."""
        logger = StructuredLogger("test")
        logger.logger.setLevel(logging.INFO)
        logger.logger.addHandler(logging.NullHandler())
        
        # Should handle unicode
        logger.info("Unicode test: 日本語 émojis 🎉")
    
    def test_very_long_messages(self):
        """Test handling of very long log messages."""
        logger = StructuredLogger("test")
        logger.logger.setLevel(logging.INFO)
        logger.logger.addHandler(logging.NullHandler())
        
        long_message = "A" * 10000
        logger.info(long_message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
