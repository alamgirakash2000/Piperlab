#!/usr/bin/env python3
"""Synchronization utilities for timestamp management."""

import time
from typing import Optional
from dataclasses import dataclass, field
import threading


@dataclass
class TimestampedData:
    """Container for data with timestamp."""
    timestamp: float
    data: any
    
    @property
    def age_ms(self) -> float:
        """Get age of data in milliseconds."""
        return (time.perf_counter() - self.timestamp) * 1000


class SyncClock:
    """High-precision synchronized clock for data capture."""
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the clock (resets to zero)."""
        with self._lock:
            self._start_time = time.perf_counter()
    
    def stop(self):
        """Stop the clock."""
        with self._lock:
            self._start_time = None
    
    def is_running(self) -> bool:
        """Check if clock is running."""
        with self._lock:
            return self._start_time is not None
    
    def get_time(self) -> float:
        """Get current time in seconds since start."""
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.perf_counter() - self._start_time
    
    def get_timestamp(self) -> float:
        """Get absolute timestamp for synchronization."""
        return time.perf_counter()


class RateController:
    """Control loop rate for consistent timing."""
    
    def __init__(self, target_hz: float):
        self.target_hz = target_hz
        self.target_period = 1.0 / target_hz
        self._last_time: Optional[float] = None
        self._overrun_count = 0
    
    def reset(self):
        """Reset the rate controller."""
        self._last_time = None
        self._overrun_count = 0
    
    def sleep(self):
        """Sleep to maintain target rate."""
        now = time.perf_counter()
        
        if self._last_time is None:
            self._last_time = now
            return
        
        elapsed = now - self._last_time
        sleep_time = self.target_period - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self._overrun_count += 1
        
        self._last_time = time.perf_counter()
    
    @property
    def overrun_count(self) -> int:
        """Get number of timing overruns."""
        return self._overrun_count


@dataclass
class SyncBuffer:
    """Thread-safe buffer for synchronized data collection."""
    max_size: int = 10000
    _data: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def append(self, timestamp: float, data: dict):
        """Append timestamped data to buffer."""
        with self._lock:
            self._data.append((timestamp, data))
            if len(self._data) > self.max_size:
                self._data.pop(0)
    
    def get_all(self) -> list:
        """Get all data and clear buffer."""
        with self._lock:
            data = self._data.copy()
            self._data.clear()
            return data
    
    def peek(self) -> Optional[tuple]:
        """Peek at the latest data without removing."""
        with self._lock:
            if self._data:
                return self._data[-1]
            return None
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
    
    def clear(self):
        """Clear all buffered data."""
        with self._lock:
            self._data.clear()
