from process_bigraph import Process


class GoldfordCRM(Process):
    config_schema = {
        "species" : "integer",
        "resources": "integer",
        "uptake_rate": "map[map[float]]",
        "byproducts": "map[map[float]]",
        "yield": "map[map[float]]",
    }

    def initialize(self, config):
        species = self.config_schema["species"]



def test_golford():
    config = {
        "species": 2,
        "resources": 2,
        "uptake_rate": {
            "sp1": {
                "resource1": 0.2,
                "resource2": 0.3},
            "sp2": {
                "resource1": 0.1,
                "resource2": 0.2}
        },
        "yield": {
            "sp1": {
                "resource1": 2.5,
                "resource2": 3.5},
            "sp2": {
                "resource1": 1.2,
                "resource2": 4.5}
        },
        "byproducts": {
            "sp1": {
                "resource1": {
                    "resource2": 0.5},
            }

    }}


if __name__ == "__main__":
    test_golford()