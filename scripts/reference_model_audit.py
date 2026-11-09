"""Verify the two explicitly named P01 bundles without mutable global aliases."""
import hashlib
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(public_bundle, second_bundle, public_pins, second_checkpoint_pin):
    public_bundle, second_bundle = Path(public_bundle), Path(second_bundle)
    assert public_bundle.resolve() != second_bundle.resolve()
    actual_public = {name: sha(public_bundle / name) for name in public_pins}
    assert actual_public == public_pins, 'Public P01 bundle differs'
    second = sha(second_bundle / 'final_model.pt')
    assert second == second_checkpoint_pin, 'Second P01 checkpoint differs'
    for name in ['model.py', 'features.py', 'config.json']:
        assert sha(second_bundle / name) == public_pins[name], name
    return dict(public_files=actual_public, second_checkpoint_sha256=second,
                second_shares_public_architecture_features_config=True)
