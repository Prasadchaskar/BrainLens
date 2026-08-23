from datetime import datetime, timedelta, timezone

DEFAULT_WINDOW_HOURS = 24

def get_current_window(
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> tuple[datetime, datetime]:
    """
    Return the UTC start and end timestamps for the
    current monitoring window.

    Example for 24 hours:

        start = now - 24 hours
        end   = now
    """

    if window_hours <= 0:
        raise ValueError(
            "window_hours must be greater than zero."
        )

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        - timedelta(
            hours=window_hours
        )
    )

    return start_time, end_time