import sys
import joblib
from coco_common.scalers import Float32StandardScaler  # noqa: F401


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python redump_model.py <old.pkl> <new.pkl>")
        raise SystemExit(1)

    old_path, new_path = sys.argv[1], sys.argv[2]

    # register class on old path
    sys.modules['__main__'].Float32StandardScaler = Float32StandardScaler

    mdl = joblib.load(old_path)
    joblib.dump(mdl, new_path)
