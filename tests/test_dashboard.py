import copy
import unittest

from tecton.dashboard import normalize_geography


class GeographyTests(unittest.TestCase):
    def setUp(self):
        self.geo = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {
            "mpio_cdpmp": "05001", "mpio_cnmbr": "Medellín", "dpto_cnmbr": "Antioquia",
        }, "geometry": {"type": "Polygon", "coordinates": [[[-75, 6], [-74, 6], [-74, 7], [-75, 6]]]}}]}

    def test_real_code_and_names_preserved(self):
        props = normalize_geography(self.geo)["features"][0]["properties"]
        self.assertEqual(props["DIVIPOLA"], "05001")
        self.assertEqual(props["municipio"], "Medellín")

    def test_reject_bad_coordinate_system_duplicate_and_unclosed_ring(self):
        duplicates = {**self.geo, "features": self.geo["features"] * 2}
        projected = copy.deepcopy(self.geo)
        projected["features"][0]["geometry"]["coordinates"][0][0] = [500000, 500000]
        unclosed = copy.deepcopy(self.geo)
        unclosed["features"][0]["geometry"]["coordinates"][0][-1] = [-73, 6]
        for geo in [duplicates, projected, unclosed]:
            with self.assertRaises(ValueError):
                normalize_geography(geo)


if __name__ == "__main__":
    unittest.main()
