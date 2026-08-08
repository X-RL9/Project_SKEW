import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from registry.base import FetchResult
from registry.ons_connector import ONSConnector, ONSDataError, parse_timeseries_observations


class ONSParserTests(unittest.TestCase):
    def test_parses_capitalized_time_and_sorts_months_chronologically(self):
        result = FetchResult(
            source_id="ons",
            fetched_at=datetime.now(timezone.utc),
            provenance_url="https://example.test",
            data={
                "observations": [
                    {"observation": "3", "dimensions": {"Time": {"id": "Mar-24"}}},
                    {"observation": "1", "dimensions": {"Time": {"id": "Jan-24"}}},
                    {"observation": "2", "dimensions": {"Time": {"id": "Feb-24"}}},
                ]
            },
        )
        time_index, values = parse_timeseries_observations(result)
        self.assertEqual(time_index.tolist(), [0, 1, 2])
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0])

    def test_sorts_two_digit_years_across_centuries(self):
        result = FetchResult(
            source_id="ons",
            fetched_at=datetime.now(timezone.utc),
            provenance_url="https://example.test",
            data={"observations": [
                {"observation": "3", "dimensions": {"Time": {"id": "Jan-04"}}},
                {"observation": "1", "dimensions": {"Time": {"id": "Jan-99"}}},
                {"observation": "2", "dimensions": {"Time": {"id": "Jan-00"}}},
            ]},
        )
        _, values = parse_timeseries_observations(result)
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0])


class ONSFetchTests(unittest.TestCase):
    def setUp(self):
        self.connector = ONSConnector()
        self.connector._resolve_edition_version = Mock(return_value=("time-series", 67))
        self.connector.get_dataset_dimensions = Mock(return_value=[
            {"name": "aggregate"}, {"name": "geography"}, {"name": "time"}
        ])

    @patch("registry.ons_connector.requests.get")
    def test_uses_live_api_name_field_and_exact_filters(self, request_get):
        response = Mock(ok=True, url="https://api.test/observations")
        response.json.return_value = {
            "observations": [
                {"observation": "1", "dimensions": {"Time": {"id": "Jan-24"}}}
            ]
        }
        request_get.return_value = response

        self.connector.fetch(
            "cpih01", aggregate="CP00", geography="K02000001", time="*"
        )
        params = request_get.call_args.kwargs["params"]
        self.assertEqual(
            params,
            {"aggregate": "CP00", "geography": "K02000001", "time": "*"},
        )

    def test_refuses_arbitrary_first_option(self):
        self.connector.get_dimension_options = Mock(return_value=[
            {"label": "Wrong series", "option": "WRONG"}
        ])
        with self.assertRaisesRegex(ONSDataError, "Refusing to select an arbitrary series"):
            self.connector.fetch("cpih01", geography="K02000001", time="*")


if __name__ == "__main__":
    unittest.main()
