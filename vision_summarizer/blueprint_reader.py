"""
Handles ALL blueprint formats: JPG/PNG, PDF, DXF/DWG.
Always returns List[PIL.Image] — one image per floor/page.
"""

from PIL import Image
import fitz              # PyMuPDF
import ezdxf
import io
import streamlit as st


# ── Image formats ──────────────────────────────────────────
def load_image(uploaded_file) -> list[Image.Image]:
    """Load a single image file and return as a one-element list."""
    img = Image.open(uploaded_file).convert("RGB")
    return [img]


# ── PDF → high-res images (300 DPI) ────────────────────────
def load_pdf(uploaded_file) -> list[Image.Image]:
    """
    Converts each PDF page to a 300 DPI PIL Image.
    300 DPI is critical — 72 DPI loses room labels.
    """
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    images = []

    for page_num, page in enumerate(doc):
        mat = fitz.Matrix(300 / 72, 300 / 72)   # 300 DPI scale
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        st.toast(f"Loaded PDF page {page_num + 1}/{len(doc)}")

    return images


# ── CAD / DXF → raster image ───────────────────────────────
def load_cad(uploaded_file) -> list[Image.Image]:
    """
    Converts DXF (and DWG via ezdxf) to a high-res PIL Image.
    Uses matplotlib backend for clean rendering.
    """
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    data = uploaded_file.read()
    try:
        doc = ezdxf.read(io.StringIO(data.decode("utf-8")))
    except Exception:
        # Binary DWG fallback — ezdxf handles most DWG versions
        doc = ezdxf.readfile(uploaded_file.name)

    msp = doc.modelspace()
    fig = plt.figure(figsize=(24, 24), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend).draw_layout(msp, finalize=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return [Image.open(buf).convert("RGB")]


# ── Master loader ───────────────────────────────────────────
def load_blueprint(uploaded_file) -> list[Image.Image]:
    """
    Routes to correct loader based on file extension.
    Always returns List[PIL.Image].
    """
    name = uploaded_file.name.lower()

    if name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
        return load_image(uploaded_file)
    elif name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    elif name.endswith((".dxf", ".dwg")):
        return load_cad(uploaded_file)
    else:
        raise ValueError(
            f"Unsupported format: {name}. "
            "Supported: JPG, PNG, PDF, DXF, DWG"
        )
