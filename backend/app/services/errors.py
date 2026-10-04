"""Service-layer errors. The API maps `status_code` straight onto the HTTP response
(JSON `{"detail": message}`), so messages are written for the end user."""
from __future__ import annotations


class ServiceError(Exception):
    status_code = 500


class InvalidInput(ServiceError):
    status_code = 400


class InvalidTicker(InvalidInput):
    """The input cannot be a ticker symbol."""


class NotFound(ServiceError):
    status_code = 404


class UnknownSymbol(NotFound):
    """Every provider positively answered "no such instrument"."""


class Unavailable(ServiceError):
    status_code = 503


class AnalysisFailed(Unavailable):
    """Synthesis itself failed (provider failures never raise; they degrade)."""
