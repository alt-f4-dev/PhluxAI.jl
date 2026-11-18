"""Bounded read-only PDF/HDF5 operations executed by a PhluxAI policy worker."""

import csv
import importlib.metadata
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile


class AnalysisError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise AnalysisError(code, message)


def integer(value, name, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum, "invalid_request", name + " must be an integer within bounds")
    require(maximum is None or value <= maximum, "limit_exceeded", name + " exceeds the configured limit")
    return value


def bounded_string(value, maximum=4096):
    text = str(value)
    require(len(text.encode("utf-8")) <= maximum, "limit_exceeded", "metadata string is too large")
    return text


def pdf_operation(path, operation, arguments, limits):
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect

    try:
        document = pdfplumber.open(path)
    except Exception as error:
        if isinstance(error, PDFPasswordIncorrect) or isinstance(error.__context__, PDFPasswordIncorrect):
            raise AnalysisError("encrypted_source", "password-protected PDFs are unsupported") from error
        raise
    with document:
        require(not document.doc.encryption, "encrypted_source", "encrypted PDFs are unsupported")
        pages = document.pages
        require(len(pages) <= limits["max_document_pages"], "limit_exceeded", "PDF page count exceeds the configured limit")
        if operation == "pdf_metadata":
            metadata = {bounded_string(key): bounded_string(value) for key, value in document.metadata.items()}
            return {"page_count": len(pages), "metadata": metadata}, [{"scheme": "pdf_document", "locator": {}}], "pdfplumber"
        selected = arguments.get("pages", [1])
        require(isinstance(selected, list) and selected, "invalid_request", "pages must be a nonempty array")
        require(len(selected) <= limits["max_pages"], "limit_exceeded", "too many PDF pages requested")
        for number in selected:
            integer(number, "page", 1)
        require(len(set(selected)) == len(selected), "invalid_request", "duplicate PDF pages")
        results, evidence = [], []
        total_cells = 0
        for number in selected:
            integer(number, "page", 1)
            require(number <= len(pages), "invalid_request", "PDF page is out of bounds")
            page = pages[number - 1]
            locator = {"page": number, "bbox": list(page.bbox), "coordinates": "PDF points, top-left origin"}
            if operation == "pdf_text":
                text = page.extract_text() or ""
                results.append({"page": number, "text": text, "has_text": bool(text)})
                evidence.append({"scheme": "pdf_page", "locator": locator})
            elif operation == "pdf_tables":
                tables = []
                for table in page.find_tables():
                    cells = table.extract()
                    total_cells += sum(len(row) for row in cells)
                    require(total_cells <= limits["max_cells"], "limit_exceeded", "table cell count exceeds the configured limit")
                    tables.append({"bbox": list(table.bbox), "cells": cells})
                    evidence.append({"scheme": "pdf_region", "locator": dict(locator, bbox=list(table.bbox))})
                results.append({"page": number, "tables": tables})
            else:
                dpi = integer(arguments.get("dpi", 150), "dpi", 72, limits["max_dpi"])
                width, height = math.ceil(page.width * dpi / 72), math.ceil(page.height * dpi / 72)
                require(width * height <= limits["max_pixels"], "limit_exceeded", "OCR raster exceeds the pixel limit")
                results.append(ocr_page(path, number, dpi, limits))
                evidence.append({"scheme": "pdf_page", "locator": dict(locator, dpi=dpi)})
        method = "poppler+tesseract" if operation == "pdf_ocr" else "pdfplumber"
        return results, evidence, method


def ocr_page(path, number, dpi, limits):
    with tempfile.TemporaryDirectory() as directory:
        base = os.path.join(directory, "page")
        try:
            subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number), "-singlefile", "-r", str(dpi),
                            "-png", path, base], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["tesseract", base + ".png", base, "-l", "eng", "tsv"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as error:
            raise AnalysisError("missing_dependency", "OCR requires Poppler and Tesseract with English language data") from error
        except subprocess.CalledProcessError as error:
            raise AnalysisError("extraction_failed", "PDF rasterization or OCR failed") from error
        result_path = pathlib.Path(base + ".tsv")
        require(result_path.stat().st_size <= limits["max_output_bytes"], "limit_exceeded", "OCR output exceeds the configured limit")
        words = []
        with result_path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row["text"].strip():
                    words.append({"text": row["text"], "confidence": float(row["conf"]),
                                  "bbox": [int(row[key]) for key in ("left", "top", "width", "height")]})
        return {"page": number, "text": " ".join(word["text"] for word in words), "words": words,
                "coordinates": "raster pixels, top-left origin", "dpi": dpi}


def hdf5_object(file, path):
    import h5py

    require(isinstance(path, str) and path.startswith("/") and "\x00" not in path,
            "invalid_request", "dataset path must be absolute within the HDF5 hierarchy")
    parts = path.split("/")[1:]
    require(path == "/" or all(part and part not in (".", "..") for part in parts),
            "invalid_request", "invalid HDF5 path components")
    group = file
    for part in ([] if path == "/" else parts):
        require(isinstance(group, h5py.Group), "invalid_request", "HDF5 path crosses a dataset")
        link = group.get(part, getlink=True)
        require(link is not None, "invalid_request", "HDF5 object does not exist")
        require(isinstance(link, h5py.HardLink), "unauthorized_reference", "soft and external HDF5 links are blocked")
        group = group[part]
    return group


def attributes(obj, limits):
    result = {}
    for count, name in enumerate(obj.attrs):
        require(count < limits["max_nodes"], "limit_exceeded", "too many attributes")
        identifier = obj.attrs.get_id(name)
        size = identifier.get_space().get_simple_extent_npoints() * identifier.dtype.itemsize
        if identifier.dtype.hasobject or size > limits["max_attribute_bytes"] or identifier.dtype.kind not in "biufSU":
            result[bounded_string(name)] = {"omitted": "variable-length, oversized, or unsupported attribute"}
            continue
        value = obj.attrs[name]
        result[bounded_string(name)] = json_value(value)
    return result


def json_value(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value] if value.ndim else json_value(value.item())
    if isinstance(value, np.generic):
        scalar = value.item()
        require(not isinstance(scalar, np.generic), "unsupported_type", "extended precision scalar is unsupported")
        return json_value(scalar)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise AnalysisError("unsupported_type", "value cannot be represented by the analysis JSON contract")


def dataset_metadata(dataset, limits):
    return {"kind": "dataset", "shape": dataset.shape, "dtype": str(dataset.dtype), "chunks": dataset.chunks,
            "compression": dataset.compression, "external_storage": bool(dataset.external),
            "virtual": dataset.is_virtual, "attributes": attributes(dataset, limits)}


def hdf5_operation(path, operation, arguments, limits):
    # Disable HDF5 plugin discovery before the library is initialized.
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    os.environ["HDF5_PLUGIN_PATH"] = ""
    import h5py
    import numpy as np

    with h5py.File(path, "r", rdcc_nbytes=min(limits["max_decoded_bytes"], 1024 * 1024)) as file:
        dataset_path = arguments.get("dataset", "/")
        obj = hdf5_object(file, dataset_path)
        if operation == "hdf5_metadata":
            data = dataset_metadata(obj, limits) if isinstance(obj, h5py.Dataset) else {"kind": "group", "attributes": attributes(obj, limits)}
        elif operation == "hdf5_hierarchy":
            require(isinstance(obj, h5py.Group), "invalid_request", "hierarchy root must be a group")
            data, pending, seen = [], [(dataset_path, obj)], set()
            while pending:
                current_path, group = pending.pop()
                address = h5py.h5o.get_info(group.id).addr
                if address in seen:
                    continue
                seen.add(address)
                for name in group:
                    require(len(data) < limits["max_nodes"], "limit_exceeded", "hierarchy exceeds the node limit")
                    full = current_path.rstrip("/") + "/" + name
                    bounded_string(full)
                    link = group.get(name, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        data.append({"path": full, "kind": "blocked_link"})
                        continue
                    child = group[name]
                    kind = "group" if isinstance(child, h5py.Group) else "dataset"
                    data.append({"path": full, "kind": kind})
                    if isinstance(child, h5py.Group):
                        pending.append((full, child))
        else:
            require(isinstance(obj, h5py.Dataset), "invalid_request", "operation requires a dataset")
            require(not obj.external and not obj.is_virtual, "unauthorized_reference", "external storage and virtual datasets are blocked")
            require(obj.dtype.kind in "biuf" and obj.dtype.itemsize <= 8 and not obj.dtype.hasobject,
                    "unsupported_type", "only fixed-width real numeric data up to 64 bits are supported")
            properties = obj.id.get_create_plist()
            for index in range(properties.get_nfilters()):
                require(properties.get_filter(index)[0] in (1, 2, 3, 4, 5, 6), "unsupported_filter", "third-party HDF5 filters are blocked")
            selection, shape = selection_for(obj.shape, arguments.get("selection"), limits)
            count = math.prod(shape) if obj.shape is not None else 0
            require(count <= limits["max_elements"], "limit_exceeded", "selection exceeds the element limit")
            require(count * obj.dtype.itemsize <= limits["max_decoded_bytes"], "limit_exceeded", "selection exceeds the decoded-byte limit")
            if obj.chunks:
                require(math.prod(obj.chunks) * obj.dtype.itemsize <= limits["max_decoded_bytes"],
                        "limit_exceeded", "decoded HDF5 chunk exceeds the byte limit")
            values = np.empty((0,), dtype=obj.dtype) if obj.shape is None else obj[selection]
            if operation == "hdf5_read":
                data = {"shape": shape, "dtype": str(obj.dtype), "values": json_value(np.asarray(values)), "elements_read": count}
            else:
                values = np.asarray(values).reshape(-1)
                finite = values[np.isfinite(values)]
                scale = float(np.max(np.abs(finite.astype(np.float64)))) if finite.size else 0.0
                mean = float(np.mean(finite.astype(np.float64) / scale) * scale) if scale else (0.0 if finite.size else None)
                data = {"count": count, "finite_count": int(finite.size), "nan_count": int(np.isnan(values).sum()),
                        "positive_infinity_count": int(np.isposinf(values).sum()), "negative_infinity_count": int(np.isneginf(values).sum()),
                        "min": json_value(finite.min()) if finite.size else None, "max": json_value(finite.max()) if finite.size else None,
                        "mean": mean}
            dataset_path = {"dataset": dataset_path, "selection": arguments.get("selection"), "selected_shape": shape}
        return data, [{"scheme": "hdf5_selection", "locator": dataset_path}], "h5py"


def selection_for(shape, supplied, limits):
    if shape is None:
        require(supplied is None or supplied == [], "invalid_request", "null datasets have no selection")
        return (), [0]
    require(len(shape) <= 32, "limit_exceeded", "dataset rank exceeds the supported limit")
    if supplied is None:
        return tuple(slice(0, length, 1) for length in shape), list(shape)
    require(isinstance(supplied, list) and len(supplied) == len(shape),
            "invalid_request", "selection must have one slice per dataset axis")
    slices, selected_shape = [], []
    for axis, item in zip(shape, supplied):
        require(isinstance(item, dict) and not (set(item) - {"start", "stop", "step"}),
                "invalid_request", "selection contains an invalid slice")
        start = integer(item.get("start", 0), "start")
        stop = integer(item.get("stop", axis), "stop")
        step = integer(item.get("step", 1), "step", 1)
        require(start <= stop <= axis, "invalid_request", "selection is outside dataset bounds")
        slices.append(slice(start, stop, step))
        selected_shape.append((stop - start + step - 1) // step)
    return tuple(slices), selected_shape


OPERATIONS = {"pdf_metadata", "pdf_text", "pdf_tables", "pdf_ocr", "hdf5_metadata", "hdf5_hierarchy", "hdf5_read", "hdf5_statistics"}


def analyze(request, path):
    operation, arguments, limits = request["operation"], request["arguments"], request["limits"]
    require(operation in OPERATIONS, "invalid_request", "unsupported analysis operation")
    allowed = {"source_ids"}
    if operation in ("pdf_text", "pdf_tables", "pdf_ocr"):
        allowed.add("pages")
    if operation == "pdf_ocr":
        allowed.add("dpi")
    if operation.startswith("hdf5_"):
        allowed.add("dataset")
    if operation in ("hdf5_read", "hdf5_statistics"):
        allowed.add("selection")
    require(not (set(arguments) - allowed), "invalid_request", "unexpected analysis arguments")
    parser = pdf_operation if operation.startswith("pdf_") else hdf5_operation
    data, evidence, method = parser(path, operation, arguments, limits)
    versions = {"python": sys.version.split()[0]}
    for package in (("pdfplumber", "pdfminer.six") if operation.startswith("pdf_") else ("h5py", "numpy")):
        versions[package] = importlib.metadata.version(package)
    if operation.startswith("hdf5_"):
        import h5py
        versions["hdf5"] = h5py.version.hdf5_version
    if operation == "pdf_ocr":
        for command, flag in (("pdftoppm", "-v"), ("tesseract", "--version")):
            version = subprocess.run([command, flag], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            versions[command] = bounded_string(version.stdout.decode("utf-8", errors="replace").splitlines()[0])
    return {"ok": True, "data": data, "evidence": evidence, "method": method, "versions": versions}


def main():
    request = json.loads(sys.argv[1])
    try:
        result = analyze(request, sys.argv[2])
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        require(len(encoded.encode("utf-8")) <= request["limits"]["max_output_bytes"],
                "limit_exceeded", "extracted result exceeds the output limit")
    except AnalysisError as error:
        encoded = json.dumps({"ok": False, "error": error.code, "message": str(error)})
    except ImportError:
        encoded = json.dumps({"ok": False, "error": "missing_dependency", "message": "analysis runtime dependencies are unavailable"})
    except Exception:
        encoded = json.dumps({"ok": False, "error": "malformed_source", "message": "source could not be parsed"})
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
