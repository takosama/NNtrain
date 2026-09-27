import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "export_qwen35_lora.py"
spec = importlib.util.spec_from_file_location("export_qwen35_lora", MODULE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture_base(path):
    payload = struct.pack("<4sIQQ", b"GGUF", 3, 1, 2)
    payload += string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    payload += string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("a") + string("b")
    payload += string("blk.0.ffn_down.weight") + struct.pack("<IQQIQ", 2, 3, 2, 0, 0)
    payload += b"\0" * (-len(payload) % 32) + struct.pack("<6f", 1, 2, 3, 4, 5, 6)
    path.write_bytes(payload)


def fixture_checkpoint(path, model, *, step=7, width=3, nonfinite=False):
    header = {"Version": 1, "ModelSha256": hashlib.sha256(model.read_bytes()).hexdigest().upper(),
              "TrainingIdentity": "dataset/config", "Step": step,
              "Options": {"Rank": 2, "Alpha": 6, "Targets": ["ffn_down"],
                          "Layers": None, "IncludeOutput": False},
              "Entries": [{"Name": "blk.0.ffn_down.weight", "Input": width, "Output": 2}]}
    encoded = json.dumps(header).encode()
    a = [1., 2., 3., 4., 5., 6.]
    b = [7., 8., 9., 10.]
    if nonfinite:
        b[1] = float("nan")
    payload = b"NNQ35LR1" + struct.pack("<i", len(encoded)) + encoded
    payload += struct.pack("<30f", *(a + b + [0.] * 20))
    path.write_bytes(payload + hashlib.sha256(payload).digest())


def read_adapter(path):
    # An independent small reader: tests directory layout, offsets and values.
    with path.open("rb") as file:
        def read(fmt):
            return struct.unpack("<" + fmt, file.read(struct.calcsize("<" + fmt)))
        def text():
            return file.read(read("Q")[0]).decode()
        magic, version, tensor_count, metadata_count = read("4sIQQ")
        assert (magic, version) == (b"GGUF", 3)
        metadata = {}
        for _ in range(metadata_count):
            name, kind = text(), read("I")[0]
            metadata[name] = text() if kind == 8 else read("f" if kind == 6 else "I")[0]
        directory = []
        for _ in range(tensor_count):
            name, ndim = text(), read("I")[0]
            shape = read("Q" * ndim)
            kind, offset = read("IQ")
            directory.append((name, shape, kind, offset))
        base_offset = (file.tell() + 31) // 32 * 32
        tensors = {}
        for name, shape, kind, offset in directory:
            assert kind == 0 and offset % 32 == 0
            file.seek(base_offset + offset)
            tensors[name] = (shape, read("f" * (shape[0] * shape[1])))
        return metadata, tensors


class ExportQwen35LoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.model, self.checkpoint, self.output = root / "base.gguf", root / "train.bin", root / "adapter.gguf"
        fixture_base(self.model)
        fixture_checkpoint(self.checkpoint, self.model)

    def export(self, **kwargs):
        return exporter.export(self.checkpoint, self.model, self.output, **kwargs)

    def test_non_square_tensor_orientation_scaling_and_optimizer_omission(self):
        result = self.export(expected_step=7)
        self.assertEqual((result["tensors"], result["rank"], result["step"]), (2, 2, 7))
        metadata, tensors = read_adapter(self.output)
        self.assertEqual(metadata["general.type"], "adapter")
        self.assertEqual(metadata["general.architecture"], "qwen35")
        self.assertEqual(metadata["adapter.type"], "lora")
        self.assertEqual(metadata["nntrain.base_model.sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest().upper())
        self.assertEqual(metadata["nntrain.training.step"], 7)
        a_shape, a = tensors["blk.0.ffn_down.weight.lora_a"]
        b_shape, b = tensors["blk.0.ffn_down.weight.lora_b"]
        self.assertEqual(a_shape, (3, 2))
        self.assertEqual(b_shape, (2, 2))
        self.assertEqual(a, (1., 2., 3., 4., 5., 6.))
        self.assertEqual(b, (7., 8., 9., 10.))
        # llama.cpp ne[0] is contiguous: B * (A * x) * alpha/rank.
        x = (2., -1., .5)
        ax = [sum(a[r * 3 + i] * x[i] for i in range(3)) for r in range(2)]
        actual = [sum(b[o * 2 + r] * ax[r] for r in range(2)) * metadata["adapter.lora.alpha"] / 2 for o in range(2)]
        self.assertEqual(actual, [175.5, 220.5])
        self.assertEqual(len(tensors), 2)

    def test_corruption_is_rejected_before_publishing(self):
        payload = bytearray(self.checkpoint.read_bytes())
        payload[-40] ^= 1
        self.checkpoint.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_wrong_base_and_wrong_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected completed step 8"):
            self.export(expected_step=8)
        with self.model.open("ab") as file:
            file.write(b"changed")
        with self.assertRaisesRegex(ValueError, "base SHA-256 mismatch"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_bad_shape_rejected_even_with_valid_checksum(self):
        fixture_checkpoint(self.checkpoint, self.model, width=4)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.export()

    def test_nonfinite_adapter_rejected_and_temporary_removed(self):
        fixture_checkpoint(self.checkpoint, self.model, nonfinite=True)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_incomplete_payload_rejected_even_with_valid_checksum(self):
        payload = self.checkpoint.read_bytes()[:-36]
        self.checkpoint.write_bytes(payload + hashlib.sha256(payload).digest())
        with self.assertRaisesRegex(ValueError, "Truncated LoRA tensor"):
            self.export()

    def test_existing_output_and_input_files_preserved(self):
        self.output.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertEqual(self.output.read_bytes(), b"keep")
        with self.assertRaisesRegex(ValueError, "distinct"):
            exporter.export(self.checkpoint, self.model, self.model, overwrite=True)
        self.export(overwrite=True)
        read_adapter(self.output)

    def test_failed_export_preserves_existing_output_with_force(self):
        self.output.write_bytes(b"keep")
        fixture_checkpoint(self.checkpoint, self.model, nonfinite=True)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.export(overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
