import unittest
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
from ui_system.components import format_last_login

class TestSessionTimezoneFormatter(unittest.TestCase):
    """Regression tests for display-only UTC -> Asia/Kolkata (IST) timezone formatter."""

    @patch("ui_system.components.get_current_ist_time")
    def test_utc_to_ist_same_day(self, mock_get_current):
        # Mock reference time (now) in IST: 2026-08-08 15:00:00
        mock_get_current.return_value = datetime(2026, 8, 8, 15, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        # Test cases for the exact same point in time (represented in different formats)
        # 05:30:00 UTC = 11:00:00 IST on 2026-08-08
        naive_utc = "2026-08-08T05:30:00"
        z_ended_utc = "2026-08-08T05:30:00Z"
        explicit_offset_utc = "2026-08-08T05:30:00+00:00"

        self.assertEqual(format_last_login(naive_utc), "Today • 11:00 AM")
        self.assertEqual(format_last_login(z_ended_utc), "Today • 11:00 AM")
        self.assertEqual(format_last_login(explicit_offset_utc), "Today • 11:00 AM")

    @patch("ui_system.components.get_current_ist_time")
    def test_utc_to_ist_explicit_offset(self, mock_get_current):
        mock_get_current.return_value = datetime(2026, 8, 8, 15, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        
        # Test offset with non-UTC (e.g. IST directly)
        explicit_ist = "2026-08-08T11:00:00+05:30"
        self.assertEqual(format_last_login(explicit_ist), "Today • 11:00 AM")

    @patch("ui_system.components.get_current_ist_time")
    def test_date_boundary_crossing_midnight(self, mock_get_current):
        # Mock reference time (now) in IST: 2026-08-08 15:00:00
        mock_get_current.return_value = datetime(2026, 8, 8, 15, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        # UTC: 2026-08-07T20:00:00Z -> IST: 2026-08-08 01:30 AM
        # This is on the date 2026-08-08 (Today) in IST, even though UTC date is 2026-08-07.
        crossing_today = "2026-08-07T20:00:00Z"
        self.assertEqual(format_last_login(crossing_today), "Today • 1:30 AM")

        # UTC: 2026-08-07T18:00:00Z -> IST: 2026-08-07 23:30 PM (Yesterday)
        crossing_yesterday = "2026-08-07T18:00:00Z"
        self.assertEqual(format_last_login(crossing_yesterday), "Yesterday • 11:30 PM")

        # UTC: 2026-08-06T18:00:00Z -> IST: 2026-08-06 23:30 PM (2 days ago)
        crossing_older = "2026-08-06T18:00:00Z"
        self.assertEqual(format_last_login(crossing_older), "06 Aug 2026 • 11:30 PM")

    def test_invalid_and_empty_inputs(self):
        # Empty string, None, or invalid strings should not crash the formatter.
        self.assertEqual(format_last_login(None), "Never")
        self.assertEqual(format_last_login(""), "Never")
        self.assertEqual(format_last_login("invalid-date-format"), "invalid-date-format")
        self.assertEqual(format_last_login("   "), "Never")

    def test_convert_utc_to_ist_same_date(self):
        # A. UTC timestamp converts to IST on the same date
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist("2026-08-08T08:03:34"), "2026-08-08T13:33:34+05:30")

    def test_convert_utc_to_ist_explicit_offset(self):
        # B. Explicit UTC offset
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist("2026-08-08T08:03:34+00:00"), "2026-08-08T13:33:34+05:30")

    def test_convert_utc_to_ist_z_suffix(self):
        # C. Z suffix
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist("2026-08-08T08:03:34Z"), "2026-08-08T13:33:34+05:30")

    def test_convert_utc_to_ist_midnight_boundary(self):
        # D. Midnight/date-boundary conversion
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist("2026-08-07T20:00:00Z"), "2026-08-08T01:30:00+05:30")

    def test_convert_utc_to_ist_invalid_empty(self):
        # E. Invalid/empty timestamp must not crash the dashboard
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist(None), None)
        self.assertEqual(convert_utc_to_ist(""), "")
        self.assertEqual(convert_utc_to_ist("invalid-date"), "invalid-date")
        self.assertEqual(convert_utc_to_ist("   "), "   ")

    def test_convert_utc_to_ist_already_aware_another_offset(self):
        # F. Already timezone-aware timestamp with another explicit offset must be converted correctly without double conversion
        from ui_system.components import convert_utc_to_ist
        self.assertEqual(convert_utc_to_ist("2026-08-08T13:33:34+05:30"), "2026-08-08T13:33:34+05:30")
        self.assertEqual(convert_utc_to_ist("2026-08-08T09:03:34+01:00"), "2026-08-08T13:33:34+05:30")

if __name__ == "__main__":
    unittest.main()

