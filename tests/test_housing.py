import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing import load_data, spatial_groups  # noqa: E402


def test_features_and_groups():
    df = load_data()
    assert len(df) == 20640
    assert (df.rooms_per_household > 0).all()
    g = spatial_groups(df)
    assert len(set(g)) > 20  # many distinct areas for spatial CV
