import json
import tempfile
from semantic_safety.semantic_router.router import SemanticRouter


def make_temp_json():
    data = [
        {
            "manipulated": "cup of water",
            "scene": "power drill",
            "families": ["liquid", "electrical"],
            "topology_template": "upward_vertical_cone",
            "weights": {
                "w_+x": 0.1,
                "w_-x": 0.1,
                "w_+y": 0.1,
                "w_-y": 0.1,
                "w_+z": 0.9,
                "w_-z": 0.0
            },
            "vertical_rule": "gravity_column",
            "lateral_decay": "moderate",
            "receptacle_attenuation": 1.0
        }
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def test_exact_hit():
    json_path = make_temp_json()
    router = SemanticRouter(json_path=json_path, persist_updates=False)
    out = router.get_risk_parameters("cup of water", "power drill")

    assert out["topology_template"] == "upward_vertical_cone"
    assert out["vertical_rule"] == "gravity_column"
    assert abs(out["weights"]["w_+z"] - 0.9) < 1e-6
    router.close()


def test_alias_hit():
    json_path = make_temp_json()
    router = SemanticRouter(json_path=json_path, persist_updates=False)
    out = router.get_risk_parameters("cup of water", "drill")

    assert out["scene"] == "power drill"
    assert out["vertical_rule"] == "gravity_column"
    router.close()


def test_unknown_fallback():
    json_path = make_temp_json()
    router = SemanticRouter(json_path=json_path, persist_updates=False)
    out = router.get_risk_parameters("cup of water", "toaster")

    assert "vertical_rule" in out
    assert "lateral_decay" in out
    assert "weights" in out
    assert out["topology_template"] in {
        "upward_vertical_cone",
        "isotropic_sphere",
        "forward_directional_cone",
        "planar_half_space",
    }
    router.close()


if __name__ == "__main__":
    test_exact_hit()
    test_alias_hit()
    test_unknown_fallback()
    print("Router tests passed.")