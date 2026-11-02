import logging

from asr_dashboard.redact import RedactingFilter, redact, register_secret


def test_redacts_patterns_and_registered_values():
    register_secret("my-very-private-key-42")
    s = redact("Authorization: Bearer abcdefghijklmnop; key=my-very-private-key-42 hf_abcdefghijklmnopq sk-1234567890abcdefghij")
    assert "abcdefghijklmnop" not in s
    assert "my-very-private-key-42" not in s
    assert "hf_abcdefghijklmnopq" not in s
    assert "sk-1234567890abcdefghij" not in s


def test_logging_filter(caplog):
    register_secret("token-zzzzzzzzzzzz")
    logger = logging.getLogger("t")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("using %s", "token-zzzzzzzzzzzz")
        try:
            raise RuntimeError("bad token-zzzzzzzzzzzz")
        except RuntimeError:
            logger.exception("failed")
    text = caplog.text
    assert "token-zzzzzzzzzzzz" not in text
