import argparse
import csv
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import torchvision
from PIL import Image


REGION_IDS = [
    "10S",
    "10T",
    "11R",
    "12R",
    "16T",
    "17R",
    "17T",
    "18S",
    "32S",
    "32T",
    "33S",
    "33T",
    "52S",
    "53S",
    "54S",
    "54T",
]

NUM_CLASSES = len(REGION_IDS)
THRESHOLD = 0.5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
VISION_MODELS_DIR = Path("/home/argus/ARGUS/Vision-Models")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ClassifierEfficient(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights=None)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.efficientnet(x))


class TrtModel:
    def __init__(self, engine_path):
        import pycuda.autoinit as cuda_autoinit
        import pycuda.driver as cuda

        self.cuda = cuda
        self.cuda_autoinit = cuda_autoinit
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.tensor_names = self._tensor_names()
        self.input_name = self._find_tensor(trt.TensorIOMode.INPUT)
        self.output_name = self._find_tensor(trt.TensorIOMode.OUTPUT)
        self.input_index = self.tensor_names.index(self.input_name)
        self.output_index = self.tensor_names.index(self.output_name)
        self.closed = False

    def _tensor_names(self):
        if hasattr(self.engine, "num_io_tensors"):
            return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        return [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

    def _tensor_mode(self, name):
        if hasattr(self.engine, "get_tensor_mode"):
            return self.engine.get_tensor_mode(name)
        index = self.tensor_names.index(name)
        return trt.TensorIOMode.INPUT if self.engine.binding_is_input(index) else trt.TensorIOMode.OUTPUT

    def _find_tensor(self, mode):
        for name in self.tensor_names:
            if self._tensor_mode(name) == mode:
                return name
        raise RuntimeError(f"No TensorRT tensor found for mode {mode}")

    def _set_input_shape(self, input_shape):
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, input_shape)
        elif hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(self.input_index, input_shape)

    def _tensor_shape(self, name, index):
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(self.context.get_tensor_shape(name))
        return tuple(self.context.get_binding_shape(index))

    @staticmethod
    def _resolve_shape(shape, batch_size):
        return tuple(batch_size if int(dim) < 0 else int(dim) for dim in shape)

    def _clear_tensor_addresses(self):
        if self.context is None or not hasattr(self.context, "set_tensor_address"):
            return

        for name in (self.input_name, self.output_name):
            try:
                self.context.set_tensor_address(name, 0)
            except Exception:
                pass

    def __call__(self, input_data):
        if self.closed:
            raise RuntimeError("TensorRT model has already been closed")

        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        self._set_input_shape(input_data.shape)

        output_shape = self._resolve_shape(
            self._tensor_shape(self.output_name, self.output_index),
            input_data.shape[0],
        )
        output_data = np.empty(output_shape, dtype=np.float32)

        input_mem = output_mem = stream = None
        try:
            input_mem = self.cuda.mem_alloc(input_data.nbytes)
            output_mem = self.cuda.mem_alloc(output_data.nbytes)
            stream = self.cuda.Stream()

            self.cuda.memcpy_htod_async(input_mem, input_data, stream)
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(self.input_name, int(input_mem))
                self.context.set_tensor_address(self.output_name, int(output_mem))
                self.context.execute_async_v3(stream_handle=stream.handle)
            else:
                bindings = [0] * len(self.tensor_names)
                bindings[self.input_index] = int(input_mem)
                bindings[self.output_index] = int(output_mem)
                self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            self.cuda.memcpy_dtoh_async(output_data, output_mem, stream)
            stream.synchronize()
        finally:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception:
                    pass
            self._clear_tensor_addresses()
            if input_mem is not None:
                try:
                    input_mem.free()
                except Exception:
                    pass
            if output_mem is not None:
                try:
                    output_mem.free()
                except Exception:
                    pass

        return output_data

    def close(self):
        if self.closed:
            return

        try:
            self.cuda.Context.synchronize()
        except Exception:
            pass
        self._clear_tensor_addresses()
        self.context = None
        self.engine = None
        self.runtime = None
        self.closed = True
        try:
            self.cuda.Context.synchronize()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def rc_weights_path(model_version):
    local_path = MODELS_DIR / "trained-rc" / model_version / "rc_model_weights.pth"
    if local_path.exists():
        return local_path

    fallback_path = VISION_MODELS_DIR / "trained-rc" / model_version / "rc_model_weights.pth"
    if fallback_path.exists():
        return fallback_path

    return local_path


def rc_trt_path(model_version):
    return MODELS_DIR / "trained-rc" / model_version / "rc_model_weights.trt"


def sample_root(sample_version):
    versioned_root = MODELS_DIR / "sample_images" / sample_version
    if versioned_root.exists():
        return versioned_root
    return MODELS_DIR / "sample_images"


def discover_samples(root):
    image_ext_priority = {".jpg": 0, ".jpeg": 1, ".png": 2}
    samples = {}

    for image_path in sorted(root.rglob("*")):
        if image_path.suffix.lower() not in image_ext_priority:
            continue

        region_id = image_path.parent.name
        if region_id not in REGION_IDS:
            continue

        key = (region_id, image_path.stem)
        previous = samples.get(key)
        if previous is None:
            samples[key] = image_path
            continue

        if image_ext_priority[image_path.suffix.lower()] < image_ext_priority[previous.suffix.lower()]:
            samples[key] = image_path

    return [(region_id, path) for (region_id, _), path in sorted(samples.items())]


def predicted_regions(scores):
    indices = np.where(scores > THRESHOLD)[0]
    return [REGION_IDS[i] for i in indices]


def format_regions(regions):
    return ", ".join(regions) if regions else "none"


def relative_to_models(path):
    try:
        return path.relative_to(MODELS_DIR)
    except ValueError:
        return path


def relative_to_base(path, base):
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def transformed_tensor_to_image(tensor):
    display_tensor = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    display_tensor = (display_tensor * std + mean).clamp(0.0, 1.0)
    display_array = (
        display_tensor.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(display_array, mode="RGB")


def save_transformed_image(tensor, image_path, rotation, output_dir):
    source_rel = relative_to_models(image_path).with_suffix("")
    output_path = output_dir / source_rel.parent / f"{source_rel.name}_rot{rotation:03d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_tensor_to_image(tensor).save(output_path)
    return output_path


def write_transformed_index(output_dir, records):
    if not records:
        return None, None

    index_path = output_dir / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    gallery_path = output_dir / "index.html"
    rows = []
    for record in records:
        image_path = escape(record["transformed_image"])
        source_image = escape(record["source_image"])
        expected_region = escape(record["expected_region"])
        pytorch_regions = escape(record["pytorch_regions"] or "none")
        tensorrt_regions = escape(record["tensorrt_regions"] or "none")
        status = "PASS" if record["passed"] else "FAIL"
        status_class = "pass" if record["passed"] else "fail"
        rows.append(
            "<tr>"
            f'<td><a href="{image_path}"><img src="{image_path}" alt="{source_image} rot {record["rotation"]}"></a></td>'
            f"<td>{source_image}</td>"
            f"<td>{record['rotation']}</td>"
            f"<td>{expected_region}</td>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{pytorch_regions}</td>"
            f"<td>{tensorrt_regions}</td>"
            f"<td>{record['tensor_min']} / {record['tensor_max']}</td>"
            f"<td>{record['tensor_mean']} / {record['tensor_std']}</td>"
            "</tr>"
        )

    gallery_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>RCNet transformed inputs</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;color:#202124;background:#fff}",
                "table{border-collapse:collapse;width:100%;font-size:13px}",
                "th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left;vertical-align:middle}",
                "th{position:sticky;top:0;background:#f8f9fa;z-index:1}",
                "img{width:112px;height:112px;object-fit:contain;background:#111}",
                ".pass{color:#137333;font-weight:700}",
                ".fail{color:#a50e0e;font-weight:700}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>RCNet transformed inputs</h1>",
                "<table>",
                "<thead><tr><th>Image</th><th>Source</th><th>Rotation</th><th>Expected</th><th>Status</th><th>PyTorch</th><th>TensorRT</th><th>Min / Max</th><th>Mean / Std</th></tr></thead>",
                "<tbody>",
                *rows,
                "</tbody></table>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return index_path, gallery_path


def score_rank(scores, region_id):
    if scores is None:
        return None

    region_index = REGION_IDS.index(region_id)
    return 1 + int(np.sum(scores > scores[region_index]))


def format_score(score):
    return "n/a" if score is None else f"{float(score):.6f}"


def default_report_dir(model_version, sample_version):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / "results" / "rc_validation" / f"{model_version}_{sample_version}_{timestamp}"


def report_path_for_record(report_dir, record):
    source_path = Path(record["source_image"])
    image_id = source_path.stem
    rotation_suffix = f"_rot{record['rotation']}" if record["rotation"] else ""
    return report_dir / f"classification_comparison_{record['expected_region']}_{image_id}{rotation_suffix}.txt"


def write_rc_report(report_dir, record):
    report_path = report_path_for_record(report_dir, record)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    expected_index = REGION_IDS.index(record["expected_region"])
    pt_scores = record["pt_scores"]
    trt_scores = record["trt_scores"]
    pt_rank = score_rank(pt_scores, record["expected_region"])
    trt_rank = score_rank(trt_scores, record["expected_region"])

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"# RC Comparison: Region {record['expected_region']} - {Path(record['source_image']).stem}\n\n")
        report_file.write(f"**Source:** {record['source_image']}\n\n")
        report_file.write(f"**Rotation:** {record['rotation']}\n\n")
        report_file.write(f"**Status:** {'PASS' if record['passed'] else 'FAIL'}\n\n")
        report_file.write("---\n\n")
        report_file.write("## Expected Region\n\n")
        report_file.write(f"- Region: {record['expected_region']}\n")
        report_file.write(f"- PyTorch score: {format_score(pt_scores[expected_index])}\n")
        report_file.write(f"- PyTorch rank: {pt_rank} of {len(REGION_IDS)}\n")
        if trt_scores is not None:
            report_file.write(f"- TensorRT score: {format_score(trt_scores[expected_index])}\n")
            report_file.write(f"- TensorRT rank: {trt_rank} of {len(REGION_IDS)}\n")
            report_file.write(f"- Max |PyTorch - TensorRT|: {format_score(record['max_diff'])}\n")
        else:
            report_file.write("- TensorRT: not run\n")

        report_file.write("\n## Predictions\n\n")
        report_file.write(f"- PyTorch: {format_regions(record['pt_regions'])}\n")
        if trt_scores is not None:
            report_file.write(f"- TensorRT: {format_regions(record['trt_regions'])}\n")
        else:
            report_file.write("- TensorRT: not run\n")

        report_file.write("\n## Scores\n\n")
        if trt_scores is not None:
            report_file.write("| Region | Expected | PT Pred | TRT Pred | PyTorch | TensorRT | Abs Diff |\n")
            report_file.write("| --- | --- | --- | --- | ---: | ---: | ---: |\n")
            for index, region_id in enumerate(REGION_IDS):
                report_file.write(
                    f"| {region_id} "
                    f"| {'yes' if region_id == record['expected_region'] else ''} "
                    f"| {'yes' if region_id in record['pt_regions'] else ''} "
                    f"| {'yes' if region_id in record['trt_regions'] else ''} "
                    f"| {format_score(pt_scores[index])} "
                    f"| {format_score(trt_scores[index])} "
                    f"| {format_score(abs(pt_scores[index] - trt_scores[index]))} |\n"
                )
        else:
            report_file.write("| Region | Expected | PT Pred | PyTorch |\n")
            report_file.write("| --- | --- | --- | ---: |\n")
            for index, region_id in enumerate(REGION_IDS):
                report_file.write(
                    f"| {region_id} "
                    f"| {'yes' if region_id == record['expected_region'] else ''} "
                    f"| {'yes' if region_id in record['pt_regions'] else ''} "
                    f"| {format_score(pt_scores[index])} |\n"
                )

    return report_path


def write_summary_report(
    report_dir,
    args,
    rotations,
    samples,
    passed,
    rotation_passed,
    rotation_failures,
    transformed_output_dir,
    report_paths,
):
    summary_path = report_dir / "validation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("# RC Validation Summary\n\n")
        summary_file.write(f"- Model version: {args.model_version}\n")
        summary_file.write(f"- Sample version: {args.sample_version}\n")
        summary_file.write(f"- Threshold: {THRESHOLD}\n")
        summary_file.write(f"- Rotations: {', '.join(str(rotation) for rotation in rotations)}\n")
        summary_file.write(f"- Sample count: {len(samples)}\n")
        summary_file.write(f"- Images passing all requested rotations: {passed}/{len(samples)}\n")
        if transformed_output_dir is not None:
            summary_file.write(f"- Transformed image folder: {transformed_output_dir}\n")

        summary_file.write("\n## Rotation Summary\n\n")
        for rotation in rotations:
            summary_file.write(f"- rot={rotation}: {rotation_passed[rotation]}/{len(samples)} passed\n")
            for failure in rotation_failures[rotation]:
                summary_file.write(f"  - failed: {failure}\n")

        summary_file.write("\n## Reports\n\n")
        for report_path in report_paths:
            summary_file.write(f"- {report_path.relative_to(report_dir)}\n")

    return summary_path


def load_pt_model(model_version, device):
    model = ClassifierEfficient().to(device)
    weights_path = rc_weights_path(model_version)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded PyTorch weights: {weights_path}")
    return model


def run_single(
    image_path,
    expected_region,
    model,
    trt_model,
    transformations,
    device,
    rotations,
    transformed_output_dir=None,
    transformed_records=None,
    validation_records=None,
):
    img = Image.open(image_path).convert("RGB")
    rotation_results = {}

    for rotation in rotations:
        rotated_img = img.rotate(rotation, expand=True) if rotation else img
        transformed_tensor = transformations(rotated_img)
        transformed_path = None
        if transformed_output_dir is not None:
            transformed_path = save_transformed_image(
                transformed_tensor,
                image_path,
                rotation,
                transformed_output_dir,
            )

        input_tensor = transformed_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pt_scores = model(input_tensor).detach().cpu().numpy().reshape(-1)
        pt_regions = predicted_regions(pt_scores)

        trt_scores = None
        trt_regions = []
        max_diff = None
        if trt_model is not None:
            trt_scores = trt_model(input_tensor.detach().cpu().numpy()).reshape(-1)
            trt_regions = predicted_regions(trt_scores)

        pt_pass = expected_region in pt_regions
        trt_pass = trt_model is None or expected_region in trt_regions
        passed = pt_pass and trt_pass
        rotation_results[rotation] = passed
        status = "PASS" if passed else "FAIL"

        rotation_label = f" rot={rotation}" if len(rotations) > 1 or rotation else ""
        print(f"[{status}] {image_path.relative_to(MODELS_DIR)}{rotation_label}")
        print(f"  expected: {expected_region}")
        print(f"  pytorch:  {format_regions(pt_regions)}")
        if trt_model is not None:
            max_diff = float(np.max(np.abs(pt_scores - trt_scores)))
            print(f"  tensorrt: {format_regions(trt_regions)}")
            print(f"  max |pt-trt|: {max_diff:.6f}")

        if validation_records is not None:
            validation_records.append(
                {
                    "source_image": str(relative_to_models(image_path)),
                    "expected_region": expected_region,
                    "rotation": rotation,
                    "passed": passed,
                    "pt_pass": pt_pass,
                    "trt_pass": trt_pass,
                    "pt_scores": pt_scores,
                    "pt_regions": pt_regions,
                    "trt_scores": trt_scores,
                    "trt_regions": trt_regions,
                    "max_diff": max_diff,
                }
            )

        if transformed_records is not None:
            tensor_stats = transformed_tensor.detach().cpu()
            transformed_rel = (
                relative_to_base(transformed_path, transformed_output_dir)
                if transformed_path is not None
                else ""
            )
            transformed_records.append(
                {
                    "source_image": str(relative_to_models(image_path)),
                    "expected_region": expected_region,
                    "rotation": rotation,
                    "transformed_image": str(transformed_rel),
                    "passed": passed,
                    "pytorch_regions": "|".join(pt_regions),
                    "tensorrt_regions": "|".join(trt_regions),
                    "tensor_min": f"{float(tensor_stats.min()):.6f}",
                    "tensor_max": f"{float(tensor_stats.max()):.6f}",
                    "tensor_mean": f"{float(tensor_stats.mean()):.6f}",
                    "tensor_std": f"{float(tensor_stats.std()):.6f}",
                }
            )

    return rotation_results


def parse_rotations(rotation_text):
    rotations = []
    for value in rotation_text.split(","):
        value = value.strip()
        if not value:
            continue
        rotations.append(int(value) % 360)
    return rotations or [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate RC model on sample images.")
    parser.add_argument("--model-version", default="V4")
    parser.add_argument("--sample-version", default="V3")
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path to write a copy of the validation output. Defaults to models/val_logs.",
    )
    parser.add_argument(
        "--rotation-augment",
        action="store_true",
        help="Run each test image at 0, 90, 180, and 270 degrees.",
    )
    parser.add_argument(
        "--rotations",
        default=None,
        help="Comma-separated rotations in degrees. Overrides --rotation-augment.",
    )
    parser.add_argument(
        "--no-trt",
        action="store_true",
        help="Skip TensorRT validation even if an engine is present.",
    )
    parser.add_argument(
        "--save-transformed-images",
        action="store_true",
        help="Save viewable PNGs of the tensors immediately before RCNet inference.",
    )
    parser.add_argument(
        "--transformed-output-dir",
        default=None,
        help="Directory for --save-transformed-images. Defaults to models/results/rc_transforms.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for RC validation text reports. Defaults to models/results/rc_validation.",
    )
    return parser.parse_args()


def default_log_path(model_version, sample_version):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / "val_logs" / f"run_sample_rc_{model_version}_{sample_version}_{timestamp}.log"


def default_transformed_output_dir(model_version, sample_version):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / "results" / "rc_transforms" / f"{model_version}_{sample_version}_{timestamp}"


def run_validation(args):
    if args.rotations is not None:
        rotations = parse_rotations(args.rotations)
    elif args.rotation_augment:
        rotations = [0, 90, 180, 270]
    else:
        rotations = [0]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pt_model(args.model_version, device)

    trt_model = None
    trt_engine_path = rc_trt_path(args.model_version)
    if args.no_trt:
        print("Skipping TensorRT validation by request")
    elif trt_engine_path.exists() and torch.cuda.is_available():
        try:
            trt_model = TrtModel(trt_engine_path)
            print(f"Loaded TensorRT engine: {trt_engine_path}")
        except Exception as exc:
            print(f"Skipping TensorRT validation: {exc}")
    elif trt_engine_path.exists():
        print(f"CUDA is not available, skipping TensorRT validation: {trt_engine_path}")
    else:
        print(f"TensorRT engine not found, running PyTorch only: {trt_engine_path}")

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    transformed_output_dir = None
    transformed_records = []
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else default_report_dir(args.model_version, args.sample_version)
    ).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing RC validation reports to: {report_dir}")

    if args.save_transformed_images or args.transformed_output_dir:
        transformed_output_dir = (
            Path(args.transformed_output_dir)
            if args.transformed_output_dir
            else default_transformed_output_dir(args.model_version, args.sample_version)
        ).resolve()
        transformed_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving transformed RCNet inputs to: {transformed_output_dir}")

    samples = discover_samples(sample_root(args.sample_version))
    print(f"Found {len(samples)} RC sample images under {sample_root(args.sample_version)}")
    if len(rotations) > 1 or rotations != [0]:
        print(f"Testing rotations: {rotations}")

    try:
        passed = 0
        rotation_passed = {rotation: 0 for rotation in rotations}
        rotation_failures = {rotation: [] for rotation in rotations}
        validation_records = []
        for expected_region, image_path in samples:
            rotation_results = run_single(
                image_path,
                expected_region,
                model,
                trt_model,
                transformations,
                device,
                rotations,
                transformed_output_dir,
                transformed_records,
                validation_records,
            )
            if all(rotation_results.values()):
                passed += 1
            for rotation, rotation_pass in rotation_results.items():
                if rotation_pass:
                    rotation_passed[rotation] += 1
                else:
                    rotation_failures[rotation].append(str(image_path.relative_to(MODELS_DIR)))

        print(f"\nSummary: {passed}/{len(samples)} sample images passed")
        print("Rotation summary:")
        for rotation in rotations:
            print(f"  rot={rotation}: {rotation_passed[rotation]}/{len(samples)} passed")
            if rotation_failures[rotation]:
                print("    failed:")
                for failure in rotation_failures[rotation]:
                    print(f"      {failure}")

        report_paths = []
        for record in validation_records:
            report_paths.append(write_rc_report(report_dir, record))
        summary_path = write_summary_report(
            report_dir,
            args,
            rotations,
            samples,
            passed,
            rotation_passed,
            rotation_failures,
            transformed_output_dir,
            report_paths,
        )
        print(f"\nRC validation summary report: {summary_path}")

        if transformed_output_dir is not None:
            csv_path, gallery_path = write_transformed_index(transformed_output_dir, transformed_records)
            print(f"\nTransformed image index: {csv_path}")
            print(f"Transformed image gallery: {gallery_path}")
    finally:
        if trt_model is not None:
            trt_model.close()


def main():
    args = parse_args()
    log_path = Path(args.log_path) if args.log_path else default_log_path(args.model_version, args.sample_version)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"Writing validation log to: {log_path}")
            run_validation(args)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
