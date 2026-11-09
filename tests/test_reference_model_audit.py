import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reference_model_audit import verify


class NamedModelBundleTests(unittest.TestCase):
    def test_distinct_checkpoints_are_checked_against_their_own_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            public, second = Path(temporary) / 'public', Path(temporary) / 'second'
            public.mkdir(); second.mkdir()
            for name in ['model.py', 'features.py', 'config.json']:
                (public / name).write_bytes(name.encode()); (second / name).write_bytes(name.encode())
            (public / 'final_model.pt').write_bytes(b'public checkpoint')
            (second / 'final_model.pt').write_bytes(b'v80 checkpoint')
            pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir()}
            second_pin = hashlib.sha256((second / 'final_model.pt').read_bytes()).hexdigest()
            self.assertEqual(verify(public, second, pins, second_pin)['public_files'], pins)
            with self.assertRaises(AssertionError):
                verify(second, public, pins, second_pin)


if __name__ == '__main__':
    unittest.main()
