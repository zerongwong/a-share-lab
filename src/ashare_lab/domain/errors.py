class AShareLabError(Exception):
    """Base error for user-facing, recoverable research failures."""


class DataUnavailableError(AShareLabError):
    """The requested dataset is unavailable and no valid cache exists."""


class DataQualityError(AShareLabError):
    """The dataset failed a safety or no-lookahead validation."""


class InsufficientHistoryError(AShareLabError):
    """There are too few observations to produce a responsible estimate."""


class FeatureDisabledError(AShareLabError):
    """A feature is deliberately disabled until its safety gate is met."""


class NotificationDeliveryError(AShareLabError):
    """A notification could not be delivered through a configured channel."""
