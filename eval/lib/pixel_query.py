"""Pixel Query: render text queries as images.

This module converts text queries to pixel images so VLMs receive the query
as a visual input instead of text tokens. This tests whether the model can
"read" the question from an image and still perform retrieval + answering.

Usage:
    renderer = PixelQueryRenderer(output_dir="pixel_queries", font_size=16, img_width=600)
    img_path = renderer.render(example_id, query_text)
"""

import logging
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _safe_stem(example_id: str) -> str:
    """Strip directory components from an id before using it in a filename.

    ``Path(x).name`` keeps only the final path component, which never contains a
    path separator and is never absolute — so a crafted id like ``../../evil`` can
    no longer escape the output directory (path traversal).
    """
    return Path(str(example_id)).name


# Default font paths (tried in order)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def _find_font(font_path: str | None = None) -> str:
    """Find a usable TrueType font on this system."""
    if font_path and os.path.exists(font_path):
        return font_path
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No suitable TTF font found. Install dejavu or liberation fonts, "
        "or pass font_path explicitly."
    )


def _wrap_text_by_pixel_width(
    text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    """Word-wrap text so each line fits within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    return lines


class PixelQueryRenderer:
    """Renders text queries as small, clear PNG images.

    Images are cached on disk so each query is only rendered once.
    """

    def __init__(
        self,
        output_dir: str = "pixel_queries",
        font_path: str | None = None,
        font_size: int = 16,
        img_width: int = 600,
        padding_x: int = 16,
        padding_y: int = 12,
        line_spacing: int = 4,
    ):
        self.output_dir = output_dir
        self.font_path = _find_font(font_path)
        self.font_size = font_size
        self.img_width = img_width
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.line_spacing = line_spacing

        os.makedirs(output_dir, exist_ok=True)
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        logger.info(
            f"PixelQueryRenderer: dir={output_dir}, font={os.path.basename(self.font_path)}, "
            f"size={font_size}, width={img_width}"
        )

    def _render_image(self, text: str) -> Image.Image:
        """Render *text* to a PIL Image (white bg, black text)."""
        max_text_width = self.img_width - 2 * self.padding_x
        lines = _wrap_text_by_pixel_width(text, self.font, max_text_width)

        # Measure line height using a reference string
        line_height = self.font.getbbox("Ay")[3] - self.font.getbbox("Ay")[1]
        total_text_height = (
            len(lines) * line_height + (len(lines) - 1) * self.line_spacing
        )
        img_height = total_text_height + 2 * self.padding_y

        img = Image.new("RGB", (self.img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = self.padding_y
        for line in lines:
            draw.text((self.padding_x, y), line, font=self.font, fill=(0, 0, 0))
            y += line_height + self.line_spacing

        return img

    def render(self, example_id: str, query_text: str) -> str:
        """Render a query and return the path to the saved PNG.

        If the image already exists on disk it is *not* re-rendered.
        """
        out_path = os.path.join(self.output_dir, f"{_safe_stem(example_id)}_query.png")
        if os.path.exists(out_path):
            return out_path

        img = self._render_image(query_text)
        img.save(out_path)
        logger.debug(f"Rendered pixel query: {out_path} ({img.size[0]}x{img.size[1]})")
        return out_path

    def render_all(self, examples: list[dict]) -> dict[str, str]:
        """Batch-render pixel queries for a list of examples.

        Args:
            examples: List of dicts with at least ``id`` and ``problem`` keys.

        Returns:
            Dict mapping example_id → pixel query image path.
        """
        id_to_path: dict[str, str] = {}
        rendered, cached = 0, 0
        for ex in examples:
            eid = ex["id"]
            path = os.path.join(self.output_dir, f"{_safe_stem(eid)}_query.png")
            if os.path.exists(path):
                cached += 1
            else:
                img = self._render_image(ex["problem"])
                img.save(path)
                rendered += 1
            id_to_path[eid] = path

        logger.info(
            f"PixelQueryRenderer: {rendered} rendered, {cached} cached, "
            f"{rendered + cached} total in {self.output_dir}"
        )
        return id_to_path


class QueryImageTextRenderer:
    """Renders query text + query image together into a single card image.

    Reuses PixelQueryRenderer for text rendering. Layout: image on top, text below
    (similar to VQA task cards). Text is centered and uses a larger font.
    Images are cached on disk.

    Usage:
        renderer = QueryImageTextRenderer(output_dir="query_cards", tiles_dir="tiles/evqa")
        path = renderer.render(example_id, query_text, query_image_path)
    """

    def __init__(
        self,
        output_dir: str = "query_cards",
        tiles_dir: str = "tiles/evqa",
        font_path: str | None = None,
        font_size: int = 22,
        card_width: int = 600,
        padding_x: int = 24,
        padding_y: int = 20,
        line_spacing: int = 6,
        image_padding: int = 16,
        text_section_padding: int = 24,
        border_radius: int = 12,
    ):
        self.output_dir = output_dir
        self.tiles_dir = tiles_dir
        self.card_width = card_width
        self.image_padding = image_padding
        self.text_section_padding = text_section_padding
        self.border_radius = border_radius

        font_path_resolved = _find_font(font_path)
        self.font_path = font_path_resolved
        self.font_size = font_size
        self.font = ImageFont.truetype(font_path_resolved, font_size)
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.line_spacing = line_spacing

        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"QueryImageTextRenderer: dir={output_dir}, card_width={card_width}, "
            f"font_size={font_size}"
        )

    def _render_query_text_centered(
        self, text: str, width: int | None = None
    ) -> Image.Image:
        """Render query text centered, reusing wrap logic from PixelQueryRenderer."""
        width = width or self.card_width
        max_text_width = width - 2 * self.padding_x

        # Calculate dynamic font size based on width
        # Ratio: width / 30 seems reasonable (600px -> 20px, 1500px -> 50px)
        dynamic_font_size = max(22, int(width / 30))

        # Use dynamic font if size is different from default
        if dynamic_font_size != self.font_size:
            try:
                font = ImageFont.truetype(self.font_path, dynamic_font_size)
            except Exception:
                font = self.font  # Fallback
        else:
            font = self.font

        lines = _wrap_text_by_pixel_width(text, font, max_text_width)

        line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        # Scale spacing proportionally
        spacing = max(4, int(self.line_spacing * (dynamic_font_size / self.font_size)))

        total_text_height = len(lines) * line_height + (len(lines) - 1) * spacing
        # Scale padding proportionally
        padding_y = max(
            self.padding_y, int(self.padding_y * (dynamic_font_size / self.font_size))
        )
        text_height = total_text_height + 2 * padding_y

        img = Image.new("RGB", (width, text_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = padding_y
        for line in lines:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = (width - line_w) // 2
            draw.text((x, y), line, font=font, fill=(50, 50, 50))
            y += line_height + spacing

        return img

    def render(
        self,
        example_id: str,
        query_text: str,
        query_image_path: str | None,
        force: bool = False,
    ) -> str:
        """Render query image + text into one card, save to disk.

        Layout: image on top, text below (same for iNaturalist and Landmarks).

        Args:
            example_id: Example identifier for filename.
            query_text: The question text to render below the image.
            query_image_path: Path to the query image (iNaturalist or Landmark photo).
                If None or file missing, only the text is rendered.
            force: If True, re-render even if output exists (e.g. when images were added later).

        Returns:
            Path to the saved PNG.
        """
        out_path = os.path.join(
            self.output_dir, f"{_safe_stem(example_id)}_query_card.png"
        )
        if os.path.exists(out_path) and not force:
            return out_path

        # Load query image
        if query_image_path and os.path.exists(query_image_path):
            try:
                query_img = Image.open(query_image_path).convert("RGB")
                # Resize if too large to ensure font size is readable and image isn't massive
                max_dim = 1536  # Standard reasonable max dimension
                if max(query_img.size) > max_dim:
                    ratio = max_dim / max(query_img.size)
                    new_size = (
                        int(query_img.width * ratio),
                        int(query_img.height * ratio),
                    )
                    query_img = query_img.resize(new_size, Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f"Failed to load query image {query_image_path}: {e}")
                query_img = None
        else:
            query_img = None

        # Card width adapts to image: expand if image is wider than card_width
        if query_img is not None:
            effective_width = max(
                self.card_width, query_img.width + 2 * self.image_padding
            )
        else:
            effective_width = self.card_width

        # Render query text at effective width
        text_img = self._render_query_text_centered(query_text, effective_width)

        # Compose: image on top, text below with padding for balanced look
        if query_img is not None:
            img_section_height = query_img.height + 2 * self.image_padding
        else:
            img_section_height = 0

        text_section_height = text_img.height + 2 * self.text_section_padding
        total_height = img_section_height + text_section_height

        card = Image.new("RGB", (effective_width, total_height), color=(255, 255, 255))
        ImageDraw.Draw(card)

        y_offset = 0
        if query_img is not None:
            x_center = (effective_width - query_img.width) // 2
            card.paste(query_img, (x_center, self.image_padding))
            y_offset = img_section_height

        # Center text block in its section
        text_y = y_offset + self.text_section_padding
        card.paste(text_img, (0, text_y))

        # Optional: rounded corners (simplified - draw white rounded rect overlay)
        if self.border_radius > 0:
            # Create mask for rounded corners
            mask = Image.new("L", card.size, 255)
            m_draw = ImageDraw.Draw(mask)
            m_draw.rounded_rectangle(
                (0, 0, card.width - 1, card.height - 1),
                radius=self.border_radius,
                fill=255,
                outline=0,
            )
            # For simple output we keep the card as-is; full rounded crop would need alpha
            pass

        card.save(out_path)
        logger.debug(f"Rendered query card: {out_path} ({card.size[0]}x{card.size[1]})")
        return out_path
