"""Small deterministic files and independent expected values for source analysis."""
import pathlib

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def make_fixtures(directory):
    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(root / "sample.pdf"), pagesize=(400, 300))
    pdf.setTitle("Analysis fixture")
    pdf.drawString(30, 260, "PhluxAI deterministic text 12345")
    pdf.linkURL("file:///etc/passwd", (30, 240, 190, 255), relative=0)
    pdf.showPage()
    for x in (30, 130, 230):
        pdf.line(x, 140, x, 240)
    for y in (140, 190, 240):
        pdf.line(30, y, 230, y)
    for x, y, text in ((50, 210, "A"), (150, 210, "B"), (50, 160, "2"), (150, 160, "4")):
        pdf.drawString(x, y, text)
    pdf.showPage()
    image = Image.new("RGB", (1200, 300), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)
    ImageDraw.Draw(image).text((40, 100), "PHLUXAI SAMPLE 12345", fill="black", font=font)
    pdf.drawImage(ImageReader(image), 0, 100, width=400, height=100)
    pdf.showPage()
    pdf.save()
    encrypted = canvas.Canvas(str(root / "encrypted.pdf"), encrypt=StandardEncryption("secret"))
    encrypted.drawString(40, 700, "Encrypted fixture")
    encrypted.save()
    canvas.Canvas(str(root / "empty.pdf")).save()
    (root / "malformed.pdf").write_bytes(b"%PDF-1.7\ninvalid")
    (root / "malformed.h5").write_bytes(b"\x89HDF\r\n\x1a\ninvalid")

    with h5py.File(root / "sample.h5", "w") as file:
        data = np.arange(60, dtype=np.float64).reshape(10, 6)
        file.create_dataset("data", data=data, chunks=(5, 3), compression="gzip")
        file["data"].attrs["units"] = np.bytes_("meV")
        file["data"].attrs["oversized"] = np.zeros(10000, dtype=np.uint8)
        file["data"].attrs["complex"] = 1 + 2j
        file.create_dataset("nonfinite", data=[1.0, 3.0, np.nan, np.inf, -np.inf])
        file.create_dataset("largefloat", data=[1e308, 1e308])
        file.create_dataset("scalar", data=np.int64(2**60 + 1))
        file.create_dataset("unsigned_scalar", data=np.uint64(2**63 + 1))
        types = file.create_group("types")
        for dtype in ("i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8", "bool", ">i4"):
            types.create_dataset(dtype, data=np.arange(24).reshape(2, 3, 4).astype(dtype))
        file.create_dataset("empty", shape=(0, 3), dtype=np.float32)
        file.create_dataset("null", data=h5py.Empty("f"))
        file.create_dataset("strings", data=np.array([b"alpha", b"beta"]))
        file.create_dataset("variable", data=["alpha", "beta"], dtype=h5py.string_dtype())
        file.create_dataset("huge", shape=(10**12,), dtype=np.float64, fillvalue=7)
        file["soft"] = h5py.SoftLink("/data")
        file["external_link"] = h5py.ExternalLink("/etc/passwd", "/")
        file.create_dataset("external_storage", shape=(1,), dtype="f8", external=[("/etc/passwd", 0, 8)])
        layout = h5py.VirtualLayout(shape=(10, 6), dtype="f8")
        layout[:] = h5py.VirtualSource("/etc/passwd", "data", shape=(10, 6))
        file.create_virtual_dataset("virtual", layout)
        group = file.create_group("cycle")
        group["self"] = group
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        plist.set_chunk((10,))
        plist.set_filter(32001, h5py.h5z.FLAG_OPTIONAL)
        h5py.h5d.create(file.id, b"unknown_filter", h5py.h5t.NATIVE_DOUBLE, h5py.h5s.create_simple((10,)), dcpl=plist)

if __name__ == "__main__":
    import sys
    make_fixtures(sys.argv[1])
