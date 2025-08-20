import pytz
from datetime import datetime, time
from typing import Optional


def get_market_timezone() -> pytz.timezone:
    """Get the market timezone (US/Eastern)."""
    return pytz.timezone('US/Eastern')


def get_current_market_time() -> datetime:
    """Get current time in market timezone."""
    return datetime.now(get_market_timezone())


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """Check if given time (or current time) is during market hours."""
    if dt is None:
        dt = get_current_market_time()
    elif dt.tzinfo is None:
        dt = get_market_timezone().localize(dt)
    elif dt.tzinfo != get_market_timezone():
        dt = dt.astimezone(get_market_timezone())
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    if dt.weekday() >= 5:  # Weekend
        return False
    
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time = dt.time()
    return market_open <= current_time <= market_close


def is_pre_market(dt: Optional[datetime] = None) -> bool:
    """Check if given time is during pre-market hours (4:00 AM - 9:30 AM ET)."""
    if dt is None:
        dt = get_current_market_time()
    elif dt.tzinfo is None:
        dt = get_market_timezone().localize(dt)
    elif dt.tzinfo != get_market_timezone():
        dt = dt.astimezone(get_market_timezone())
    
    if dt.weekday() >= 5:  # Weekend
        return False
    
    pre_market_start = time(4, 0)
    market_open = time(9, 30)
    
    current_time = dt.time()
    return pre_market_start <= current_time < market_open


def is_after_hours(dt: Optional[datetime] = None) -> bool:
    """Check if given time is during after-hours (4:00 PM - 8:00 PM ET)."""
    if dt is None:
        dt = get_current_market_time()
    elif dt.tzinfo is None:
        dt = get_market_timezone().localize(dt)
    elif dt.tzinfo != get_market_timezone():
        dt = dt.astimezone(get_market_timezone())
    
    if dt.weekday() >= 5:  # Weekend
        return False
    
    market_close = time(16, 0)
    after_hours_end = time(20, 0)
    
    current_time = dt.time()
    return market_close < current_time <= after_hours_end


def get_next_market_open() -> datetime:
    """Get the next market open time."""
    now = get_current_market_time()
    market_open_time = time(9, 30)
    
    # If it's already past market open today and market is open, return tomorrow
    if now.time() >= market_open_time and now.weekday() < 5:
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        # Add days until next weekday
        days_ahead = 1
        if now.weekday() == 4:  # Friday
            days_ahead = 3  # Skip to Monday
        next_open = next_open.replace(day=now.day + days_ahead)
    else:
        # Market hasn't opened today, or it's weekend
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If it's weekend, move to Monday
        if now.weekday() >= 5:  # Weekend
            days_ahead = 7 - now.weekday()  # Days until Monday
            next_open = next_open.replace(day=now.day + days_ahead)
    
    return next_open


def format_market_time(dt: datetime) -> str:
    """Format datetime for market timezone display."""
    if dt.tzinfo is None:
        dt = get_market_timezone().localize(dt)
    elif dt.tzinfo != get_market_timezone():
        dt = dt.astimezone(get_market_timezone())
    
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')