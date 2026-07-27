"""Screenshot utilities using Selenium WebDriver.

Selenium and webdriver_manager are optional — only needed when capturing
screenshots (--url-screenshot mode). Import is deferred to function scope
so the eval script works without them for --local-api users.
"""

import base64
import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)


def setup_driver(window_width=1024, window_height=2000, device_scale_factor=1):
    """Set up Chrome WebDriver.

    Args:
        window_width: Viewport width (1024 = tile width, ensures screenshots align with tile grid).
        window_height: Initial viewport height.
        device_scale_factor: Pixel density (1 = standard, 2 = retina quality).
    """
    import shutil

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    snap_chromedriver = "/snap/bin/chromium.chromedriver"
    if os.path.exists(snap_chromedriver):
        driver_path = snap_chromedriver
    elif shutil.which("chromedriver"):
        driver_path = shutil.which("chromedriver")
    else:
        driver_path = ChromeDriverManager().install()
    service = Service(driver_path)
    options = webdriver.ChromeOptions()

    # Find Chrome binary path
    chrome_binary = None
    for chrome_path in [
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    ]:
        if os.path.exists(chrome_path):
            chrome_binary = chrome_path
            break

    if chrome_binary:
        options.binary_location = chrome_binary

    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--window-size={window_width},{window_height}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--no-zygote")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    )
    # Retina-quality rendering (2x pixel density)
    if device_scale_factor and device_scale_factor > 1:
        options.add_argument(f"--force-device-scale-factor={device_scale_factor}")
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def _capture_with_scroll(driver, output_path, scroll_pause=0.8, max_scrolls=100):
    """Capture full page by scrolling and stitching screenshots.

    Works for PDF viewers, infinite scroll pages, and other dynamic content.
    Uses image comparison to detect when scrolling has stopped.
    """
    import hashlib
    import tempfile

    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys

    def get_screenshot_hash(path):
        """Get hash of screenshot to detect changes."""
        with Image.open(path) as img:
            return hashlib.md5(img.tobytes()).hexdigest()

    viewport_height = driver.execute_script("return window.innerHeight")
    viewport_width = driver.execute_script("return window.innerWidth")

    # Try to scroll to top using multiple methods
    driver.execute_script("window.scrollTo(0, 0)")
    actions = ActionChains(driver)
    actions.send_keys(Keys.HOME)
    actions.perform()
    time.sleep(scroll_pause)

    screenshots = []
    last_hash = None
    scroll_count = 0
    consecutive_same = 0

    while scroll_count < max_scrolls:
        # Take screenshot
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        driver.save_screenshot(temp_file.name)

        # Check if image changed (detect end of scrolling)
        current_hash = get_screenshot_hash(temp_file.name)

        if current_hash == last_hash:
            consecutive_same += 1
            os.unlink(temp_file.name)
            if consecutive_same >= 2:
                # Scrolling stopped, we've reached the end
                break
        else:
            consecutive_same = 0
            screenshots.append(temp_file.name)
            last_hash = current_hash

        # Scroll down using CDP mouse wheel event (works for PDF viewers)
        center_x = viewport_width // 2
        center_y = viewport_height // 2
        try:
            driver.execute_cdp_cmd(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": center_x,
                    "y": center_y,
                    "deltaX": 0,
                    "deltaY": int(viewport_height * 0.8),  # Scroll 80% of viewport
                },
            )
        except Exception:
            # Fallback to ActionChains
            actions = ActionChains(driver)
            actions.send_keys(Keys.PAGE_DOWN)
            actions.perform()

        time.sleep(scroll_pause)
        scroll_count += 1

    if not screenshots:
        driver.save_screenshot(output_path)
        return

    if len(screenshots) == 1:
        # Only one screenshot, just use it
        os.rename(screenshots[0], output_path)
        return

    # Stitch screenshots vertically
    # Each screenshot is viewport_height, but they overlap
    # We'll stack them with some overlap detection
    images = [Image.open(p) for p in screenshots]

    # Simple stacking: assume each Page Down scrolls ~80% of viewport
    overlap = int(viewport_height * 0.2)
    total_height = viewport_height + (len(images) - 1) * (viewport_height - overlap)

    stitched = Image.new("RGB", (viewport_width, total_height), (255, 255, 255))

    y_offset = 0
    for i, img in enumerate(images):
        if i == 0:
            stitched.paste(img, (0, 0))
            y_offset = viewport_height - overlap
        else:
            # Crop top overlap region and paste
            cropped = img.crop((0, overlap, viewport_width, viewport_height))
            stitched.paste(cropped, (0, y_offset))
            y_offset += viewport_height - overlap

    # Close images and clean up
    for img in images:
        img.close()
    for p in screenshots:
        if os.path.exists(p):
            os.unlink(p)

    # Trim any white space at bottom
    stitched = stitched.crop((0, 0, viewport_width, y_offset + overlap))
    stitched.save(output_path)


def _eager_load_images(driver):
    """Force lazy images to load by promoting data-src and setting loading='eager'."""
    driver.execute_script("""
        (function() {
            var imgs = document.querySelectorAll('img');
            for (var i = 0; i < imgs.length; i++) {
                var img = imgs[i];
                try {
                    if (img.loading === 'lazy') img.loading = 'eager';
                    var dataSrc = img.getAttribute('data-src') || (img.dataset && img.dataset.src);
                    var dataSrcset = img.getAttribute('data-srcset') || (img.dataset && img.dataset.srcset);
                    if (dataSrc) img.setAttribute('src', dataSrc);
                    if (dataSrcset) img.setAttribute('srcset', dataSrcset);
                } catch(e) {}
            }
        })();
    """)


def _wait_for_images(driver, timeout=10):
    """Wait for all document images to finish loading (load or error)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = driver.execute_script("""
            return Array.from(document.images || [])
                .filter(function(img) { return !(img.complete && img.naturalWidth > 0); }).length;
        """)
        if pending == 0:
            return
        time.sleep(0.3)


def _scroll_to_trigger_lazy_load(driver, page_height):
    """Scroll through the page to trigger lazy-loaded content, then scroll back to top."""
    viewport_height = driver.execute_script("return window.innerHeight") or 1080
    y = 0
    while y < page_height:
        driver.execute_script(f"window.scrollTo(0, {y})")
        time.sleep(0.15)
        _eager_load_images(driver)
        y += viewport_height
    # Wait for all images to load after full scroll
    _wait_for_images(driver, timeout=10)
    # Scroll back to top
    driver.execute_script("window.scrollTo(0, 0)")
    time.sleep(0.3)


def capture_screenshot(url, output_path, full_page=False, scroll_capture=False):
    """Capture screenshot of a URL.

    Args:
        url: URL to capture
        output_path: Path to save screenshot
        full_page: If True, resize window to capture full page (works for normal pages)
        scroll_capture: If True, scroll and stitch screenshots (works for PDF viewers, etc.)
    """
    driver = None
    try:
        driver = setup_driver()
        driver.get(url)
        time.sleep(3)  # Wait for initial load

        # Force lazy images to load eagerly
        _eager_load_images(driver)
        _wait_for_images(driver, timeout=5)

        if scroll_capture:
            # Scroll-based capture for PDF viewers and similar
            # just for PDF
            _capture_with_scroll(driver, output_path)
        elif full_page:
            # Get page height, keep original window width to avoid horizontal tiling
            total_height = driver.execute_script("return document.body.scrollHeight")
            current_window = driver.get_window_size()

            # Scroll through page to trigger lazy-loaded images
            _scroll_to_trigger_lazy_load(driver, total_height)

            # Re-measure height (may change after lazy content loads)
            total_height = driver.execute_script("return document.body.scrollHeight")
            driver.set_window_size(current_window["width"], total_height)
            time.sleep(0.5)

            # Final wait for any images triggered by resize
            _eager_load_images(driver)
            _wait_for_images(driver, timeout=5)

            driver.save_screenshot(output_path)
        else:
            driver.save_screenshot(output_path)

        # Convert to RGB (remove alpha channel if present)
        with Image.open(output_path) as img:
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[3])
                img = bg
                img.save(output_path)
            elif img.mode != "RGB":
                img = img.convert("RGB")
                img.save(output_path)

        return True
    except Exception as e:
        print(f"Screenshot failed for {url}: {e}")
        return False
    finally:
        if driver:
            driver.quit()


def encode_image(image_path, max_pixels: int = 150_000_000, max_height: int = 8000):
    """Encode image to base64, compressing if too large.

    Used for Vector DB retrieval where consistent image sizes help with embedding.

    Args:
        image_path: Path to image file.
        max_pixels: Maximum pixels allowed (default 150M).
        max_height: Maximum height in pixels (default 8000).
    """
    if not os.path.exists(image_path):
        return None

    try:
        # Increase PIL limit temporarily
        Image.MAX_IMAGE_PIXELS = 300_000_000

        with Image.open(image_path) as img:
            # Check if compression needed
            total_pixels = img.width * img.height
            needs_compression = total_pixels > max_pixels or img.height > max_height

            if not needs_compression:
                # Just read and encode directly
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            # Compress: resize to fit within limits
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Calculate new size
            if img.height > max_height:
                ratio = max_height / img.height
                new_width = int(img.width * ratio)
                new_height = max_height
            else:
                # Scale down to fit max_pixels
                ratio = (max_pixels / total_pixels) ** 0.5
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Encode to JPEG
            from io import BytesIO

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Failed to encode image {image_path}: {e}")
        return None


def encode_image_for_vlm(image_path, max_pixels: int = 89_000_000):
    """Encode image to base64 for VLM ground truth, minimal processing.

    For Ground Truth evaluation, we want to preserve original image quality
    and let the VLM handle resizing according to its own requirements.
    Only applies PIL safety limit (89M pixels).

    Args:
        image_path: Path to image file.
        max_pixels: Maximum pixels (default 89M, PIL's safety limit).
    """
    if not os.path.exists(image_path):
        return None

    try:
        # Increase PIL limit temporarily
        Image.MAX_IMAGE_PIXELS = 300_000_000

        with Image.open(image_path) as img:
            total_pixels = img.width * img.height

            # Only compress if exceeds PIL safety limit
            if total_pixels <= max_pixels:
                # Just read and encode directly - no resize
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            # Compress only if exceeds max_pixels
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Scale down to fit max_pixels
            ratio = (max_pixels / total_pixels) ** 0.5
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Encode to JPEG
            from io import BytesIO

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Failed to encode image {image_path}: {e}")
        return None
