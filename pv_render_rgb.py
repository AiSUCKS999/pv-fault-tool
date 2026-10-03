# RGB terrain, PV compositing, panel realism, and RGB fault drawing.
def terrain_layer(location_id):
    location = location_by_id(location_id)
    base = load_rgb(location["source"], location["crop"])

    if location.get("prebuilt"):
        terrain = base
    elif location_id == "grass_open":
        original = load_rgb("scenario_1_og_pv.jpeg")
        blurred = original.filter(ImageFilter.GaussianBlur(location["blur"]))
        terrain = original.copy()
        terrain.paste(blurred, (0, 0), panel_mask())
    else:
        terrain = base.filter(ImageFilter.GaussianBlur(location["blur"]))
        terrain = terrain.filter(ImageFilter.MedianFilter(7))

    contrast, color, brightness = location["color"]
    terrain = ImageEnhance.Contrast(terrain).enhance(contrast)
    terrain = ImageEnhance.Color(terrain).enhance(color)
    terrain = ImageEnhance.Brightness(terrain).enhance(brightness)
    return terrain


def pv_cutout():
    original = load_rgb("scenario_1_og_pv.jpeg")
    cutout = Image.new("RGBA", BASE_SIZE, (0, 0, 0, 0))
    cutout.paste(original.convert("RGBA"), (0, 0), panel_mask())
    return cutout


def local_rect_mask(size, radius=1, feather=0.45):
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    radius = max(0, min(radius, width // 2, height // 2))
    if radius:
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    else:
        draw.rectangle((0, 0, width - 1, height - 1), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask


def _blend_disc(canvas, cx, cy, radius, target, alpha):
    h, w = canvas.shape[:2]
    radius = max(1, int(radius))
    x1 = max(0, int(cx) - radius - 1)
    y1 = max(0, int(cy) - radius - 1)
    x2 = min(w, int(cx) + radius + 2)
    y2 = min(h, int(cy) + radius + 2)
    if x2 <= x1 or y2 <= y1:
        return

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.circle(mask, (int(cx) - x1, int(cy) - y1), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), max(0.35, radius * 0.35))
    local_alpha = np.clip(mask * alpha, 0, 1)
    if canvas.ndim == 2:
        canvas[y1:y2, x1:x2] = canvas[y1:y2, x1:x2] * (1 - local_alpha) + float(target) * local_alpha
    else:
        canvas[y1:y2, x1:x2] = (
            canvas[y1:y2, x1:x2] * (1 - local_alpha[..., None])
            + np.array(target, dtype=np.float32) * local_alpha[..., None]
        )


def _blend_point(canvas, cx, cy, target, alpha):
    h, w = canvas.shape[:2]
    x = int(round(cx))
    y = int(round(cy))
    if not (0 <= x < w and 0 <= y < h):
        return
    if canvas.ndim == 2:
        canvas[y, x] = canvas[y, x] * (1 - alpha) + float(target) * alpha
    else:
        canvas[y, x] = canvas[y, x] * (1 - alpha) + np.array(target, dtype=np.float32) * alpha


def _blend_mask(canvas, mask, target, alpha):
    if mask.size == 0:
        return
    local_alpha = np.clip(mask.astype(np.float32) * alpha, 0, 1)
    if canvas.ndim == 2:
        canvas[:] = canvas * (1 - local_alpha) + float(target) * local_alpha
    else:
        canvas[:] = canvas * (1 - local_alpha[..., None]) + np.array(target, dtype=np.float32) * local_alpha[..., None]


def _blend_local_mask(canvas, x1, y1, mask, target, alpha):
    h, w = canvas.shape[:2]
    x1 = int(x1)
    y1 = int(y1)
    x2 = min(w, x1 + mask.shape[1])
    y2 = min(h, y1 + mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    local_mask = np.clip(mask[: y2 - y1, : x2 - x1].astype(np.float32) * alpha, 0, 1)
    if canvas.ndim == 2:
        canvas[y1:y2, x1:x2] = canvas[y1:y2, x1:x2] * (1 - local_mask) + float(target) * local_mask
    else:
        canvas[y1:y2, x1:x2] = (
            canvas[y1:y2, x1:x2] * (1 - local_mask[..., None])
            + np.array(target, dtype=np.float32) * local_mask[..., None]
        )


def _blend_line(canvas, p1, p2, target, alpha, width=1, blur=0.25):
    h, w = canvas.shape[:2]
    pad = max(3, int(round(width + blur * 4 + 2)))
    x1 = max(0, min(int(round(p1[0])), int(round(p2[0]))) - pad)
    y1 = max(0, min(int(round(p1[1])), int(round(p2[1]))) - pad)
    x2 = min(w, max(int(round(p1[0])), int(round(p2[0]))) + pad + 1)
    y2 = min(h, max(int(round(p1[1])), int(round(p2[1]))) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.line(
        mask,
        (int(round(p1[0])) - x1, int(round(p1[1])) - y1),
        (int(round(p2[0])) - x1, int(round(p2[1])) - y1),
        1.0,
        max(1, int(width)),
        cv2.LINE_AA,
    )
    if blur:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    _blend_local_mask(canvas, x1, y1, mask, target, alpha)


def _blend_diamond(canvas, cx, cy, radius, target, alpha, blur=0.25):
    radius = max(1, int(round(radius)))
    h, w = canvas.shape[:2]
    pad = max(2, int(round(radius + blur * 4 + 2)))
    x1 = max(0, int(round(cx)) - pad)
    y1 = max(0, int(round(cy)) - pad)
    x2 = min(w, int(round(cx)) + pad + 1)
    y2 = min(h, int(round(cy)) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return
    points = np.array(
        [
            [int(round(cx)) - x1, int(round(cy - radius)) - y1],
            [int(round(cx + radius)) - x1, int(round(cy)) - y1],
            [int(round(cx)) - x1, int(round(cy + radius)) - y1],
            [int(round(cx - radius)) - x1, int(round(cy)) - y1],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.fillConvexPoly(mask, points, 1.0, cv2.LINE_AA)
    if blur:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    _blend_local_mask(canvas, x1, y1, mask, target, alpha)


def draw_cell_string_dot_texture(surface, ix1, iy1, ix2, iy2, seed, thermal=False, panel_min=150, strength=1.0, nodes_only=False):
    """Dense, consistent contact-dot lattice across each PV module surface."""
    if nodes_only:
        return

    cell_w = max(1, (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
    cell_h = max(1, (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS)
    if cell_w < 5 or cell_h < 7:
        return

    if thermal:
        dark_target = max(18, panel_min - 86)
        dark_alpha = min(0.42, 0.42 * strength)
    else:
        dark_target = (28, 42, 54)
        dark_alpha = 0.026 * strength

    spacing_x = max(3, int(round(cell_w * 0.22)))
    spacing_y = max(3, int(round(cell_h * 0.15)))
    margin_x = max(2, int(round(cell_w * 0.14)))
    margin_y = max(2, int(round(cell_h * 0.10)))

    xs = list(range(int(ix1 + margin_x), int(ix2 - margin_x) + 1, spacing_x))
    ys = list(range(int(iy1 + margin_y), int(iy2 - margin_y) + 1, spacing_y))
    if not xs or not ys:
        return

    # Center the lattice so every scenario gets the same manufactured, non-random geometry.
    used_w = (len(xs) - 1) * spacing_x
    used_h = (len(ys) - 1) * spacing_y
    start_x = int(round((ix1 + ix2 - used_w) / 2))
    start_y = int(round((iy1 + iy2 - used_h) / 2))
    xs = [start_x + idx * spacing_x for idx in range(len(xs))]
    ys = [start_y + idx * spacing_y for idx in range(len(ys))]

    for px in xs:
        for py in ys:
            _blend_point(surface, px, py, dark_target, dark_alpha)


def visual_square_lattice_counts(ix1, iy1, ix2, iy2):
    """Choose square-ish visual tile counts without changing selectable cells."""
    width = max(1, ix2 - ix1)
    height = max(1, iy2 - iy1)
    pitch = float(np.clip(min(width, height) / 3.15, 4.0, 9.0))
    cols = int(np.clip(round(width / pitch), 5, 14))
    rows = int(np.clip(round(height / pitch), 3, 9))
    return cols, rows


def draw_projected_square_lattice(surface, quad, ix1, iy1, ix2, iy2, seed, thermal=False, panel_min=150):
    """Draw the visible cell-tile lattice seen in real UAV thermal module imagery."""
    cols, rows = visual_square_lattice_counts(ix1, iy1, ix2, iy2)
    if cols < 2 or rows < 2:
        return

    if thermal:
        value = max(8, panel_min - 74)
        width = 1
        opacity = 0.72
        blur = 0.38
    else:
        value = (36, 50, 60)
        width = 1
        opacity = 0.20
        blur = 0.30

    for idx in range(1, cols):
        u = idx / cols
        p1 = tuple(np.round(quad_point(quad, u, 0.02)).astype(int))
        p2 = tuple(np.round(quad_point(quad, u, 0.98)).astype(int))
        blend_wavy_line(surface, p1, p2, value, width, seed_for(seed, "visual-lattice-v", idx), 0.10, opacity, blur)
    for idx in range(1, rows):
        v = idx / rows
        p1 = tuple(np.round(quad_point(quad, 0.02, v)).astype(int))
        p2 = tuple(np.round(quad_point(quad, 0.98, v)).astype(int))
        blend_wavy_line(surface, p1, p2, value, width, seed_for(seed, "visual-lattice-h", idx), 0.10, opacity, blur)


def source_panel_crop(source, source_row, left_ratio, right_ratio):
    sx1 = source_row["x1"] + 2
    sy1 = source_row["y1"] + 2
    sx2 = source_row["x2"] - 2
    sy2 = source_row["y2"] - 2
    sw = max(1, sx2 - sx1)
    left = int(sx1 + sw * left_ratio)
    right = int(sx1 + sw * right_ratio)
    if right <= left:
        right = min(sx2, left + 1)
    return source.crop((left, sy1, right, sy2))


def source_module_crop(source, source_row, module_number, module_count):
    sx1 = source_row["x1"] + 2
    sy1 = source_row["y1"] + 2
    sx2 = source_row["x2"] - 2
    sy2 = source_row["y2"] - 2
    sw = max(1, sx2 - sx1)
    source_index = (module_number - 1) % max(1, module_count)
    left = int(sx1 + sw * source_index / max(1, module_count))
    right = int(sx1 + sw * (source_index + 1) / max(1, module_count))
    pad = max(1, int((right - left) * 0.04))
    crop = source.crop((max(sx1, left + pad), sy1, min(sx2, right - pad), sy2)).convert("RGB")
    arr = np.array(crop)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] < 118) & (hsv[:, :, 1] > 18)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) > max(8, crop.width * crop.height * 0.05):
        x1 = clamp(int(xs.min()) - 1, 0, crop.width - 1)
        x2 = clamp(int(xs.max()) + 2, x1 + 1, crop.width)
        y1 = clamp(int(ys.min()) - 1, 0, crop.height - 1)
        y2 = clamp(int(ys.max()) + 2, y1 + 1, crop.height)
        crop = crop.crop((x1, y1, x2, y2))
    return crop


def clamp(value, low, high):
    return max(low, min(high, value))


def mean_luma(image):
    return ImageStat.Stat(ImageOps.grayscale(image)).mean[0]


def add_panel_grain(patch, seed, strength=0.035):
    rng = random.Random(seed)
    noise = Image.effect_noise(patch.size, rng.uniform(5.0, 9.0)).convert("L")
    noise = noise.point(lambda value: int(128 + (value - 128) * strength))
    return ImageChops.multiply(patch.convert("RGB"), noise.convert("RGB")).convert("RGBA")


def match_patch_lighting(patch, terrain, box):
    x1, y1, x2, y2 = box
    terrain_crop = terrain.crop((x1, y1, x2, y2)).convert("RGB")
    terrain_level = mean_luma(terrain_crop)
    patch_level = max(1, mean_luma(patch.convert("RGB")))
    # Let the real background lighting influence the pasted PVs without washing
    # out the original panel texture.
    brightness = clamp((terrain_level / patch_level) * 0.72 + 0.34, 0.76, 1.18)
    contrast = clamp(0.92 + (terrain_level / 255) * 0.18, 0.92, 1.08)
    patch = ImageEnhance.Brightness(patch).enhance(brightness)
    patch = ImageEnhance.Contrast(patch).enhance(contrast)
    return patch


def terrain_texture_angle(terrain_crop):
    gray = np.array(ImageOps.grayscale(terrain_crop), dtype=np.uint8)
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = 0.5 * math.atan2(float((2 * gx * gy).mean()), float((gx * gx - gy * gy).mean()) + 1e-6)
    return angle


def harmonize_patch_texture(rgb, terrain_crop, seed, edge_strength=0.11):
    """Borrow local terrain grain/illumination so pasted PVs sit in the scene."""
    width, height = rgb.size
    if width < 3 or height < 3:
        return rgb

    rng = np.random.default_rng(seed)
    panel = np.array(rgb.convert("RGB"), dtype=np.float32)
    terrain_arr = np.array(terrain_crop.resize((width, height), Image.Resampling.BICUBIC).convert("RGB"), dtype=np.float32)
    terrain_gray = cv2.cvtColor(terrain_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    low_sigma = max(2.0, min(width, height) * 0.22)
    low = cv2.GaussianBlur(terrain_gray, (0, 0), low_sigma)
    mid = terrain_gray - cv2.GaussianBlur(terrain_gray, (0, 0), max(0.8, min(width, height) * 0.05))
    low_norm = (low - low.mean()) / (low.std() + 1e-6)
    mid_norm = (mid - mid.mean()) / (mid.std() + 1e-6)

    modulation = 1.0 + low_norm[..., None] * 0.025 + mid_norm[..., None] * 0.018
    sensor_grain = rng.normal(0, 1.0, (height, width, 1)).astype(np.float32)
    sensor_grain = cv2.GaussianBlur(sensor_grain, (0, 0), 0.55)
    if sensor_grain.ndim == 2:
        sensor_grain = sensor_grain[..., None]
    modulation += sensor_grain * 0.006
    panel *= modulation

    # Subtle terrain-color spill on the cut edges prevents sticker-like borders.
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    edge = np.clip(1.0 - dist / max(2.0, min(width, height) * 0.18), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.85)[..., None] * edge_strength
    panel = panel * (1.0 - edge) + terrain_arr * edge

    # Directional low-amplitude streaks follow the local terrain orientation.
    angle = terrain_texture_angle(terrain_crop)
    axis = np.cos(angle) * xx + np.sin(angle) * yy
    streak = np.sin(axis / max(3.0, min(width, height) * 0.42) + seed % 17)
    panel += streak[..., None] * 1.25

    return Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8), "RGB")


def mute_bright_panel_lines(rgb):
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright = gray > np.percentile(gray, 98)
    target = np.array([36, 46, 58], dtype=np.float32)
    arr[bright] = arr[bright] * 0.92 + target * 0.08
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def procedural_panel_module(size, seed):
    width, height = size
    scale = 4
    hi_w = max(32, int(width * scale))
    hi_h = max(24, int(height * scale))
    rng = np.random.default_rng(seed)

    module = np.zeros((hi_h, hi_w, 3), dtype=np.float32)
    module[:] = np.array([22, 35, 55], dtype=np.float32)

    frame = max(3, int(min(hi_w, hi_h) * 0.035))
    cv2.rectangle(module, (0, 0), (hi_w - 1, hi_h - 1), (132, 142, 146), frame)
    cv2.rectangle(module, (frame, frame), (hi_w - frame - 1, hi_h - frame - 1), (17, 30, 50), -1)

    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    gap = max(2, int(min(hi_w, hi_h) * 0.018))
    inner_x1 = frame + gap
    inner_y1 = frame + gap
    inner_x2 = hi_w - frame - gap
    inner_y2 = hi_h - frame - gap
    inner_w = max(1, inner_x2 - inner_x1)
    inner_h = max(1, inner_y2 - inner_y1)

    yy, xx = np.mgrid[0:hi_h, 0:hi_w]
    glass_gradient = 1.08 - 0.16 * (yy / max(1, hi_h - 1)) + 0.04 * (xx / max(1, hi_w - 1))

    for row in range(rows):
        for col in range(cols):
            x1 = int(inner_x1 + col * inner_w / cols) + gap // 2
            y1 = int(inner_y1 + row * inner_h / rows) + gap // 2
            x2 = int(inner_x1 + (col + 1) * inner_w / cols) - gap // 2
            y2 = int(inner_y1 + (row + 1) * inner_h / rows) - gap // 2
            if x2 <= x1 or y2 <= y1:
                continue
            cell_tint = np.array(
                [
                    18 + rng.integers(-4, 5),
                    38 + rng.integers(-5, 7),
                    70 + rng.integers(-7, 9),
                ],
                dtype=np.float32,
            )
            cv2.rectangle(module, (x1, y1), (x2, y2), tuple(int(v) for v in cell_tint), -1)
            bus_alpha = 0.62
            for lx in np.linspace(x1 + (x2 - x1) * 0.22, x2 - (x2 - x1) * 0.22, 3):
                cv2.line(module, (int(lx), y1 + 1), (int(lx), y2 - 1), (156, 172, 184), max(1, hi_w // 360))
            if y2 - y1 > 12:
                cv2.line(module, (x1 + 1, (y1 + y2) // 2), (x2 - 1, (y1 + y2) // 2), (118, 136, 154), 1)
            # soften busbars back into the silicon so they do not become UI lines
            module[y1:y2, x1:x2] = module[y1:y2, x1:x2] * (1 - bus_alpha * 0.08) + cell_tint * (bus_alpha * 0.08)

    draw_cell_string_dot_texture(
        module,
        inner_x1,
        inner_y1,
        inner_x2,
        inner_y2,
        seed_for(seed, "module-string-dot-texture"),
        thermal=False,
        strength=0.85,
    )

    # Real glass texture: mild sensor grain plus diagonal reflected light.
    noise = rng.normal(0, 3.2, (hi_h, hi_w, 1)).astype(np.float32)
    diagonal = np.sin((xx + yy * 0.34 + seed % 41) / max(8, hi_w * 0.18))[..., None] * 3.5
    module = module * glass_gradient[..., None] + noise + diagonal

    glare = np.zeros((hi_h, hi_w, 1), dtype=np.float32)
    glare_y = int(hi_h * (0.22 + 0.18 * ((seed % 17) / 16)))
    cv2.line(glare, (0, glare_y), (hi_w - 1, max(0, glare_y - hi_h // 6)), 1.0, max(2, hi_h // 14))
    glare = cv2.GaussianBlur(glare, (0, 0), max(1.0, hi_h * 0.035))
    if glare.ndim == 2:
        glare = glare[..., None]
    module += glare * 18

    module = np.clip(module, 0, 255).astype(np.uint8)
    panel = Image.fromarray(module, "RGB").resize((width, height), Image.Resampling.LANCZOS)
    return ImageEnhance.Sharpness(panel).enhance(1.35)


def color_match_array_to_background(source_rgb, target_rgb):
    if source_rgb.size == 0 or target_rgb.size == 0:
        return source_rgb
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = cv2.meanStdDev(src_lab)
    bg_mean, bg_std = cv2.meanStdDev(bg_lab)
    matched = src_lab.copy()
    src_l_mean = float(src_mean[0][0])
    src_l_std = float(src_std[0][0]) + 1e-5
    bg_l_mean = float(bg_mean[0][0])
    bg_l_std = float(bg_std[0][0])
    target_l_mean = clamp(bg_l_mean * 0.54, 42, 108)
    target_l_std = clamp(bg_l_std * 0.72, 12, 34)
    matched_l = (src_lab[:, :, 0] - src_l_mean) * (target_l_std / src_l_std) + target_l_mean
    matched[:, :, 0] = matched_l * 0.76 + src_lab[:, :, 0] * 0.24
    # Keep silicon blue/black; borrow only a tiny ambient color cast.
    matched[:, :, 1] = src_lab[:, :, 1] * 0.92 + float(bg_mean[1][0]) * 0.08
    matched[:, :, 2] = src_lab[:, :, 2] * 0.92 + float(bg_mean[2][0]) * 0.08
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def smooth_unit_field(shape, seed, sigma, octaves=2):
    """Deterministic 0..1 smooth field for cloud/sun and sensor variation."""
    height, width = shape
    rng = np.random.default_rng(seed)
    field = np.zeros((height, width), dtype=np.float32)
    weight_total = 0.0
    for octave in range(octaves):
        weight = 1.0 / (octave + 1)
        local_sigma = max(0.7, sigma / (octave + 1))
        noise = rng.normal(0, 1, (height, width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), local_sigma)
        field += noise * weight
        weight_total += weight
    field /= max(weight_total, 1e-6)
    field -= field.min()
    field /= field.max() + 1e-6
    return field


def pv_environment_multiplier(shape, seed, strength=0.10):
    """Low-frequency irradiance/cloud layer shared by PV RGB and thermal rendering."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sun_angle = ((seed % 137) / 137.0) * math.pi
    axis = np.cos(sun_angle) * xx / max(1, width - 1) + np.sin(sun_angle) * yy / max(1, height - 1)
    axis = (axis - axis.min()) / (axis.max() - axis.min() + 1e-6)
    broad_cloud = smooth_unit_field(shape, seed_for(seed, "cloud-field"), max(9.0, min(width, height) * 0.75), octaves=3)
    fine_cloud = smooth_unit_field(shape, seed_for(seed, "fine-cloud-field"), max(3.0, min(width, height) * 0.22), octaves=2)
    field = (broad_cloud - 0.5) * strength + (fine_cloud - 0.5) * strength * 0.35 + (axis - 0.5) * strength * 0.55
    return np.clip(1.0 + field, 1.0 - strength * 0.95, 1.0 + strength * 0.95).astype(np.float32)


def pv_cell_temperature_field(shape, seed, module_count=1, strength=7.0):
    """Panel-internal thermal texture: cell mismatch, busbar bands, and edge cooling."""
    height, width = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = (smooth_unit_field(shape, seed_for(seed, "cell-thermal-low"), max(2.0, min(width, height) * 0.24), 3) - 0.5) * strength
    field += (smooth_unit_field(shape, seed_for(seed, "cell-thermal-mid"), max(1.0, min(width, height) * 0.075), 2) - 0.5) * strength * 0.55

    # Module-to-module offsets keep neighboring rectangles from sharing one flat hue.
    layouts, _, _ = module_layouts_for_dimensions(width, height, max(1, module_count))
    for layout in layouts:
        ix1, iy1, ix2, iy2 = layout["inner"]
        offset = rng.normal(0, strength * 0.22)
        field[iy1 : iy2 + 1, ix1 : ix2 + 1] += offset
        local_gradient = np.linspace(
            rng.normal(-strength * 0.16, strength * 0.04),
            rng.normal(strength * 0.16, strength * 0.04),
            max(1, ix2 - ix1 + 1),
            dtype=np.float32,
        )[None, :]
        field[iy1 : iy2 + 1, ix1 : ix2 + 1] += local_gradient
        for col in range(CUSTOM_MODULE_CELL_COLS):
            cx1 = int(ix1 + col * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
            cx2 = int(ix1 + (col + 1) * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
            cell_offset = rng.normal(0, strength * 0.16)
            field[iy1 : iy2 + 1, cx1 : cx2 + 1] += cell_offset

    edge_dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy])
    edge_cooling = np.clip(1.0 - edge_dist / max(2.0, min(width, height) * 0.22), 0, 1)
    field -= cv2.GaussianBlur(edge_cooling, (0, 0), 1.0) * strength * 0.34
    return field.astype(np.float32)


def apply_panel_rgb_realism(panel, seed, location_id, module_count):
    """Add cloud/sun/cell hue spread without changing the controlled layout."""
    height, width = panel.shape[:2]
    arr = panel.astype(np.float32)
    env = pv_environment_multiplier((height, width), seed_for(location_id, seed, "rgb-cloud"), strength=0.075)
    cell_field = pv_cell_temperature_field((height, width), seed_for(location_id, seed, "rgb-cell-field"), module_count, strength=5.5)
    arr *= env[..., None]
    # Warmer cells get a little red/yellow lift; cooler cells stay blue.
    warm = np.clip(cell_field / 12.0, -1, 1)[..., None]
    arr += np.concatenate([warm * 6.0, warm * 2.5, -warm * 3.5], axis=2)
    sensor = np.random.default_rng(seed_for(seed, "rgb-sensor-grain")).normal(0, 1.1, (height, width, 1)).astype(np.float32)
    sensor = cv2.GaussianBlur(sensor, (0, 0), 0.45)
    if sensor.ndim == 2:
        sensor = sensor[..., None]
    arr += sensor
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_wavy_line(image, p1, p2, color, width, seed, amplitude=0.75):
    """Draw slightly imperfect seams so module/cell borders do not look computer-perfect."""
    rng = np.random.default_rng(seed)
    x1, y1 = p1
    x2, y2 = p2
    steps = max(3, int(math.hypot(x2 - x1, y2 - y1) / 6))
    points = []
    for idx in range(steps + 1):
        t = idx / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        jitter = rng.normal(0, amplitude)
        if abs(x2 - x1) > abs(y2 - y1):
            y += jitter
        else:
            x += jitter
        points.append((int(round(x)), int(round(y))))
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, width, cv2.LINE_AA)


def blend_wavy_line(image, p1, p2, value, width, seed, amplitude=0.75, opacity=0.45, blur=0.55):
    """Blend a wavy seam into an image rather than stamping a hard UI line."""
    if image.size == 0:
        return
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    draw_wavy_line(mask, p1, p2, 255, max(1, width), seed, amplitude)
    mask = cv2.GaussianBlur(mask, (0, 0), blur).astype(np.float32) / 255.0
    mask *= float(opacity)
    if image.ndim == 2:
        image[:] = image * (1.0 - mask) + float(value) * mask
    else:
        color = np.array(value, dtype=np.float32).reshape(1, 1, 3)
        image[:] = image.astype(np.float32) * (1.0 - mask[..., None]) + color * mask[..., None]


THERMAL_SCENE_SETTINGS = {
    "far": {
        "sensor_scale": 0.52,
        "sensor_blur": 1.20,
        "detail_strength": 0.55,
        "grid_strength": 0.55,
        "grid_width": 1,
        "seam_temp_offset": -13.0,
    },
    "close": {
        "sensor_scale": 0.86,
        "sensor_blur": 0.48,
        "detail_strength": 1.00,
        "grid_strength": 1.00,
        "grid_width": 1,
        "seam_temp_offset": -20.0,
    },
}


def thermal_scene_settings(scene_distance):
    value = "close" if scene_distance is None else str(scene_distance).strip().lower()
    if value in {"far", "wide", "distant", "overview", "long"}:
        return THERMAL_SCENE_SETTINGS["far"]
    return THERMAL_SCENE_SETTINGS["close"]


def draw_thermal_module_cell_seams(temp, row, module_count, panel_min, include_string_grid=True, scene_distance="close"):
    """Thermal module/cell geometry aligned to the actual module quadrilateral."""
    if temp is None:
        return

    settings = thermal_scene_settings(scene_distance)
    grid_strength = float(settings["grid_strength"])
    seam_width = max(1, int(settings["grid_width"]))
    module_count = max(1, int(module_count))
    cols = max(1, int(CUSTOM_MODULE_CELL_COLS))
    rows = max(1, int(CUSTOM_MODULE_CELL_ROWS))
    seam_value = float(panel_min + float(settings["seam_temp_offset"]) * grid_strength)
    frame_value = float(panel_min - 44.0 * grid_strength)

    if not include_string_grid:
        cols = 1
        rows = 1

    def bilinear_point(corners, u, v):
        tl, tr, br, bl = corners
        top = tl * (1.0 - u) + tr * u
        bottom = bl * (1.0 - u) + br * u
        return top * (1.0 - v) + bottom * v

    def draw_segment(p1, p2, value, width):
        h, w = temp.shape[:2]
        x1 = int(round(float(p1[0])))
        y1 = int(round(float(p1[1])))
        x2 = int(round(float(p2[0])))
        y2 = int(round(float(p2[1])))
        if max(x1, x2) < 0 or min(x1, x2) >= w or max(y1, y2) < 0 or min(y1, y2) >= h:
            return
        cv2.line(temp, (x1, y1), (x2, y2), float(value), max(1, width), cv2.LINE_AA)

    for layout in module_layouts_for_row(row, module_count):
        outer = np.asarray(module_visual_polygon(row, layout, "outer"), dtype=np.float32)
        if outer.shape[0] < 4:
            continue
        if outer.shape[0] > 4:
            x_min = float(np.min(outer[:, 0]))
            x_max = float(np.max(outer[:, 0]))
            y_min = float(np.min(outer[:, 1]))
            y_max = float(np.max(outer[:, 1]))
            outer = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        else:
            outer = outer[:4]

        cv2.polylines(temp, [np.round(outer).astype(np.int32)], True, frame_value, seam_width, cv2.LINE_AA)
        if not include_string_grid:
            continue

        for col in range(1, cols):
            u = col / float(cols)
            draw_segment(bilinear_point(outer, u, 0.0), bilinear_point(outer, u, 1.0), seam_value, seam_width)

        for row_idx in range(1, rows):
            v = row_idx / float(rows)
            draw_segment(bilinear_point(outer, 0.0, v), bilinear_point(outer, 1.0, v), seam_value, seam_width)

        string_count = max(1, min(cols, 6))
        for string_idx in range(1, string_count):
            u = string_idx / float(string_count)
            draw_segment(bilinear_point(outer, u, 0.0), bilinear_point(outer, u, 1.0), seam_value - 3.0, seam_width)


def jittered_module_polygon(box, seed, max_jitter=1.2):
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = box
    jitter = lambda: float(rng.uniform(-max_jitter, max_jitter))
    return np.array(
        [
            [int(round(x1 + jitter())), int(round(y1 + jitter()))],
            [int(round(x2 + jitter())), int(round(y1 + jitter()))],
            [int(round(x2 + jitter())), int(round(y2 + jitter()))],
            [int(round(x1 + jitter())), int(round(y2 + jitter()))],
        ],
        dtype=np.int32,
    )


def module_visual_polygon(row, layout, variant="outer"):
    """Use only slight perspective because the background is a near-nadir UAV view."""
    ox1, oy1, ox2, oy2 = layout[variant]
    w = max(1, ox2 - ox1)
    h = max(1, oy2 - oy1)
    rng = np.random.default_rng(seed_for(row["id"], layout["module_number"], "visual-module-polygon", variant))

    # Keep the plan-view footprint essentially rectangular; only a faint foreshortening
    # is appropriate for a near-vertical camera looking at slightly tilted modules.
    tilt_deg = float(rng.uniform(6.0, 12.0))
    projected_depth_loss = h * (1.0 - np.cos(np.radians(tilt_deg)))
    top_lift = float(np.clip(projected_depth_loss * 0.70, 0.0, max(0.25, h * 0.035)))
    bottom_drop = float(np.clip(projected_depth_loss * 0.08, 0.0, max(0.15, h * 0.008)))
    inset = float(rng.uniform(0.0, max(0.12, w * 0.0015)))
    perspective_inset = float(np.clip(w * 0.008, 0.25, 1.5))

    pts = np.array(
        [
            [ox1 + inset + perspective_inset, oy1 + top_lift],
            [ox2 - inset - perspective_inset, oy1 + top_lift],
            [ox2 - inset, oy2 + bottom_drop],
            [ox1 + inset, oy2 + bottom_drop],
        ],
        dtype=np.float32,
    )
    pts[:, 0] = np.clip(pts[:, 0], 0, BASE_SIZE[0] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, BASE_SIZE[1] - 1)
    return pts

def module_visual_polygon_local(row, layout, variant="outer"):
    pts = module_visual_polygon(row, layout, variant)
    ox1, oy1, _, _ = layout[variant]
    local = pts.copy()
    local[:, 0] -= ox1
    local[:, 1] -= oy1
    return local


def quad_point(quad, u, v):
    top = quad[0] * (1 - u) + quad[1] * u
    bottom = quad[3] * (1 - u) + quad[2] * u
    return top * (1 - v) + bottom * v


def warp_rgba_to_polygon(patch, polygon, canvas_size=BASE_SIZE):
    src = np.array(patch.convert("RGBA"))
    ph, pw = src.shape[:2]
    if pw < 2 or ph < 2:
        return np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
    src_quad = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src_quad, polygon.astype(np.float32))
    return cv2.warpPerspective(
        src,
        transform,
        canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_composite_rgba_array(base_rgb, overlay_rgba):
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    if not np.any(alpha > 0):
        return base_rgb
    base = base_rgb.astype(np.float32)
    over = overlay_rgba[:, :, :3].astype(np.float32)
    return np.clip(base * (1 - alpha) + over * alpha, 0, 255).astype(np.uint8)


def procedural_panel_row(size, module_count, seed, location_id=None, source_textures=None):
    width, height = size
    scale = 4
    layouts, hi_w, hi_h = module_layouts_for_dimensions(width, height, module_count, scale=scale, high_res=True)
    rng = np.random.default_rng(seed)
    model_textures = actual_panel_model_textures()
    using_actual_model = bool(model_textures)
    textures = list(model_textures if using_actual_model else [])
    row = np.zeros((hi_h, hi_w, 3), dtype=np.float32)
    row[:] = np.array([16, 28, 48], dtype=np.float32)

    yy, xx = np.mgrid[0:hi_h, 0:hi_w]
    global_glass = 1.05 - 0.13 * (yy / max(1, hi_h - 1)) + 0.035 * (xx / max(1, hi_w - 1))

    for layout in layouts:
        mx1, _, mx2, _ = layout["outer"]
        ox1, oy1, ox2, oy2 = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        if mx2 - mx1 < 6:
            continue
        frame = max(1, ix1 - mx1)
        if not using_actual_model:
            cv2.rectangle(row, (mx1, 0), (mx2, hi_h - 1), (44, 54, 64), max(1, frame))
        glass_tint = np.array(
            [
                13 + rng.integers(-2, 3),
                29 + rng.integers(-3, 4),
                52 + rng.integers(-4, 5),
            ],
            dtype=np.float32,
        )
        if textures:
            if using_actual_model:
                texture = textures[0]
            elif source_textures and layout["module_number"] <= len(source_textures):
                texture = textures[layout["module_number"] - 1]
            else:
                texture = textures[int(rng.integers(0, len(textures)))]
            th, tw = texture.shape[:2]
            crop_pad_x = 0 if using_actual_model else max(0, tw // 22)
            crop_pad_y = 0 if using_actual_model else max(0, th // 22)
            tx1 = int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0
            ty1 = int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0
            tx2 = tw - (int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0)
            ty2 = th - (int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0)
            texture_crop = texture[ty1:max(ty1 + 1, ty2), tx1:max(tx1 + 1, tx2)]
            paste_x1, paste_y1, paste_x2, paste_y2 = (ox1, oy1, ox2, oy2) if using_actual_model else (ix1, iy1, ix2, iy2)
            textured = cv2.resize(
                texture_crop,
                (max(1, paste_x2 - paste_x1 + 1), max(1, paste_y2 - paste_y1 + 1)),
                interpolation=cv2.INTER_LANCZOS4,
            ).astype(np.float32)
            # Keep the real panel crop dominant. Only nudge tone enough to
            # harmonize repeated crops with the scene and avoid clone-like rows.
            panel_mean = textured.reshape(-1, 3).mean(axis=0)
            cool_target = np.array([40, 62, 82], dtype=np.float32) + rng.normal(0, 5, 3).astype(np.float32)
            textured = textured + (cool_target - panel_mean) * (0.035 if using_actual_model else 0.10)
            env = pv_environment_multiplier(
                textured.shape[:2],
                seed_for(seed, layout["module_number"], "real-module-env"),
                strength=0.025 if using_actual_model else 0.055,
            )
            textured *= env[..., None]
            texture_noise = rng.normal(0, 0.25 if using_actual_model else 0.65, textured.shape).astype(np.float32)
            target_h = max(0, min(hi_h, paste_y2 + 1) - paste_y1)
            target_w = max(0, min(hi_w, paste_x2 + 1) - paste_x1)
            if target_h > 0 and target_w > 0:
                row[paste_y1 : paste_y1 + target_h, paste_x1 : paste_x1 + target_w] = np.clip(
                    textured[:target_h, :target_w] + texture_noise[:target_h, :target_w],
                    0,
                    255,
                )
            if using_actual_model:
                draw_cell_string_dot_texture(
                    row,
                    ix1,
                    iy1,
                    ix2,
                    iy2,
                    seed_for(seed, layout["module_number"], "row-string-dot-texture"),
                    thermal=False,
                    strength=0.52,
                )
                continue
        else:
            cv2.rectangle(row, (ix1, iy1), (ix2, iy2), tuple(int(v) for v in glass_tint), -1)

        # Render the real 6x2 PV cell structure, but as low-contrast glass
        # seams. The bright selectable overlay remains only in the zoom view.
        cell_cols = CUSTOM_MODULE_CELL_COLS
        cell_rows = CUSTOM_MODULE_CELL_ROWS
        seam_width = max(1, hi_w // 1100)
        for cell_row in range(cell_rows):
            for cell_col in range(cell_cols):
                cx1 = int(ix1 + cell_col * (ix2 - ix1) / cell_cols) + 1
                cx2 = int(ix1 + (cell_col + 1) * (ix2 - ix1) / cell_cols) - 1
                cy1 = int(iy1 + cell_row * (iy2 - iy1) / cell_rows) + 1
                cy2 = int(iy1 + (cell_row + 1) * (iy2 - iy1) / cell_rows) - 1
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                if not textures:
                    cell_tint = glass_tint + np.array(
                        [
                            rng.integers(-1, 2),
                            rng.integers(-2, 3),
                            rng.integers(-3, 4),
                        ],
                        dtype=np.float32,
                    )
                    cv2.rectangle(row, (cx1, cy1), (cx2, cy2), tuple(int(v) for v in np.clip(cell_tint, 0, 255)), -1)

        draw_cell_string_dot_texture(
            row,
            ix1,
            iy1,
            ix2,
            iy2,
            seed_for(seed, layout["module_number"], "row-string-dot-texture"),
            thermal=False,
            strength=0.78 if textures else 1.0,
        )

        seam_color_v = (38, 56, 68) if textures else (72, 96, 110)
        seam_color_h = (34, 50, 62) if textures else (66, 88, 102)
        for idx in range(1, cell_cols):
            x = int(ix1 + idx * (ix2 - ix1) / cell_cols)
            cv2.line(row, (x, iy1 + 1), (x, iy2 - 1), seam_color_v, seam_width, cv2.LINE_AA)
        for idx in range(1, cell_rows):
            y = int(iy1 + idx * (iy2 - iy1) / cell_rows)
            cv2.line(row, (ix1 + 1, y), (ix2 - 1, y), seam_color_h, seam_width, cv2.LINE_AA)

    noise = rng.normal(0, 1.7, (hi_h, hi_w, 1)).astype(np.float32)
    row = row * global_glass[..., None] + noise
    row = np.clip(row, 0, 255).astype(np.uint8)
    row = apply_panel_rgb_realism(row, seed_for(seed, "hires-realism"), location_id, module_count)
    row = cv2.resize(row, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return row


def real_texture_thermal_panel_row(size, module_count, seed, location_id, panel_min, panel_max):
    width, height = size
    rng = np.random.default_rng(seed)
    model_textures = actual_panel_model_textures()
    using_actual_model = bool(model_textures)
    textures = model_textures if using_actual_model else real_panel_textures(texture_group_for_location(location_id))
    row = np.full((height, width), panel_min - 28, dtype=np.float32)
    layouts, _, _ = module_layouts_for_dimensions(width, height, max(1, module_count))
    module_mid = (panel_min + panel_max) * 0.5

    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        cv2.rectangle(row, (ox1, oy1), (ox2, oy2), max(8, panel_min - 82), -1)
        paste_x1, paste_y1, paste_x2, paste_y2 = (ox1, oy1, ox2, oy2) if using_actual_model else (ix1, iy1, ix2, iy2)
        local_w = max(1, paste_x2 - paste_x1 + 1)
        local_h = max(1, paste_y2 - paste_y1 + 1)

        if textures:
            texture = textures[0] if using_actual_model else textures[int(rng.integers(0, len(textures)))]
            th, tw = texture.shape[:2]
            crop_pad_x = 0 if using_actual_model else max(0, tw // 24)
            crop_pad_y = 0 if using_actual_model else max(0, th // 24)
            tx1 = int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0
            ty1 = int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0
            tx2 = tw - (int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0)
            ty2 = th - (int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0)
            crop = texture[ty1:max(ty1 + 1, ty2), tx1:max(tx1 + 1, tx2)]
            crop = cv2.resize(crop, (local_w, local_h), interpolation=cv2.INTER_LANCZOS4)
            gray_patch = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            smooth = cv2.GaussianBlur(gray_patch, (0, 0), max(0.55, min(local_w, local_h) * 0.035))
            low, high = np.percentile(smooth, [8, 92])
            norm = np.clip((smooth - low) / max(1.0, high - low), 0, 1)
            thermal = module_mid + (norm - 0.5) * (panel_max - panel_min) * 0.38

            # RGB module photos usually show metallic/cell separators as bright.
            # In thermal imagery those physical seams are cooler, so invert the
            # separator evidence instead of making it hot.
            seam_raw = np.clip((gray_patch - np.percentile(gray_patch, 68)) / max(1.0, np.percentile(gray_patch, 96) - np.percentile(gray_patch, 68)), 0, 1)
            edge = cv2.Laplacian(gray_patch, cv2.CV_32F)
            edge = np.clip(np.abs(edge) / (np.percentile(np.abs(edge), 92) + 1e-6), 0, 1)
            seam_mask = cv2.GaussianBlur(np.maximum(seam_raw * 0.85, edge * 0.45), (0, 0), 0.55)
            thermal -= seam_mask * (22 + 0.12 * (panel_max - panel_min))
        else:
            thermal = np.full((local_h, local_w), module_mid, dtype=np.float32)

        cell_variation = pv_cell_temperature_field(
            (local_h, local_w),
            seed_for(seed, layout["module_number"], "real-texture-cell-thermal"),
            1,
            strength=3.0,
        )
        thermal += cell_variation
        thermal *= pv_environment_multiplier(
            thermal.shape,
            seed_for(seed, layout["module_number"], "real-texture-module-env"),
            strength=0.045,
        )
        thermal += rng.normal(0, 0.85, thermal.shape).astype(np.float32)
        target_h = max(0, min(height, paste_y2 + 1) - paste_y1)
        target_w = max(0, min(width, paste_x2 + 1) - paste_x1)
        if target_h > 0 and target_w > 0:
            row[paste_y1 : paste_y1 + target_h, paste_x1 : paste_x1 + target_w] = np.clip(
                thermal[:target_h, :target_w],
                panel_min - 28,
                panel_max + 6,
            )
        draw_cell_string_dot_texture(
            row,
            ix1,
            iy1,
            ix2,
            iy2,
            seed_for(seed, layout["module_number"], "real-texture-thermal-string-grid"),
            thermal=True,
            panel_min=panel_min,
            strength=0.90,
        )

        # Controlled 6x2 target grid remains present, but as a physical thermal
        # seam laid over the real crop rather than a drawn annotation.
        if using_actual_model:
            continue
        for idx in range(1, CUSTOM_MODULE_CELL_COLS):
            x = int(ix1 + idx * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
            blend_wavy_line(row, (x, iy1), (x, iy2), max(10, panel_min - 42), 1, seed_for(seed, layout["module_number"], "thermal-real-v", idx), 0.22, 0.24, 0.60)
        for idx in range(1, CUSTOM_MODULE_CELL_ROWS):
            y = int(iy1 + idx * (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS)
            blend_wavy_line(row, (ix1, y), (ix2, y), max(10, panel_min - 40), 1, seed_for(seed, layout["module_number"], "thermal-real-h", idx), 0.20, 0.22, 0.60)

    return cv2.GaussianBlur(row, (0, 0), 0.28).astype(np.float32)


def procedural_thermal_panel_row(size, module_count, seed, panel_min, panel_max):
    width, height = size
    rng = np.random.default_rng(seed)
    row = np.full((height, width), panel_min - 34, dtype=np.float32)
    layouts, _, _ = module_layouts_for_dimensions(width, height, max(1, module_count))

    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        if ox2 <= ox1 or oy2 <= oy1 or ix2 <= ix1 or iy2 <= iy1:
            continue

        module_mid = rng.uniform(panel_min + 9, panel_max + 4)
        frame_temp = max(4, panel_min - 112)
        separator_temp = max(8, panel_min - 82)
        rim_temp = max(10, panel_min - 64)
        frame_w = max(2, min(5, int(round(max(1, oy2 - oy1) * 0.13))))
        cv2.rectangle(row, (ox1, oy1), (ox2, oy2), frame_temp, -1)
        cv2.rectangle(row, (ix1, iy1), (ix2, iy2), module_mid, -1)

        cell_w = max(2, (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
        cell_h = max(2, (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS)
        for cy in range(CUSTOM_MODULE_CELL_ROWS):
            for cx in range(CUSTOM_MODULE_CELL_COLS):
                x1 = int(round(ix1 + cx * cell_w))
                x2 = int(round(ix1 + (cx + 1) * cell_w))
                y1 = int(round(iy1 + cy * cell_h))
                y2 = int(round(iy1 + (cy + 1) * cell_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                cell_temp = module_mid + rng.normal(0, 1.1)
                cv2.rectangle(row, (x1, y1), (x2, y2), cell_temp, -1)

        draw_cell_string_dot_texture(
            row,
            ix1,
            iy1,
            ix2,
            iy2,
            seed_for(seed, layout["module_number"], "procedural-thermal-string-grid"),
            thermal=True,
            panel_min=panel_min,
            strength=0.90,
        )

        for idx in range(1, CUSTOM_MODULE_CELL_COLS):
            x = int(round(ix1 + idx * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS))
            cv2.line(row, (x, iy1), (x, iy2), separator_temp, 1, cv2.LINE_AA)
        for idx in range(1, CUSTOM_MODULE_CELL_ROWS):
            y = int(round(iy1 + idx * (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS))
            cv2.line(row, (ix1, y), (ix2, y), separator_temp, 1, cv2.LINE_AA)
        cv2.rectangle(row, (ox1, oy1), (ox2, oy2), frame_temp, frame_w, cv2.LINE_AA)
        cv2.rectangle(row, (ix1, iy1), (ix2, iy2), rim_temp, 1, cv2.LINE_AA)

    cell_field = pv_cell_temperature_field((height, width), seed_for(seed, "old-procedural-field"), module_count, strength=2.1)
    row += cell_field
    row *= pv_environment_multiplier((height, width), seed_for(seed, "old-procedural-env"), strength=0.025)
    return cv2.GaussianBlur(row, (0, 0), 0.22).astype(np.float32)


def redraw_visible_module_geometry(panel, module_count):
    """Restore faint physical seams after ambient color matching."""
    height, width = panel.shape[:2]
    layouts, _, _ = module_layouts_for_dimensions(width, height, module_count)
    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        cv2.rectangle(panel, (ox1, oy1), (ox2, oy2), (54, 62, 66), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (ix1, iy1), (ix2, iy2), (38, 48, 58), 1, cv2.LINE_AA)
        for idx in range(1, CUSTOM_MODULE_CELL_COLS):
            x = int(ix1 + idx * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
            cv2.line(panel, (x, iy1), (x, iy2), (36, 52, 62), 1, cv2.LINE_AA)
        for idx in range(1, CUSTOM_MODULE_CELL_ROWS):
            y = int(iy1 + idx * (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS)
            cv2.line(panel, (ix1, y), (ix2, y), (34, 48, 58), 1, cv2.LINE_AA)
    return panel


def apply_mounting_hardware(output_rgb, row, module_count, seed, location_id=None):
    """Draw location-appropriate racking without turning the scene into a diagram."""
    if location_id in {"floating_water", "rooftop_warehouse"}:
        return output_rgb
    layouts = module_layouts_for_row(row, module_count)
    if not layouts:
        return output_rgb

    h, w = output_rgb.shape[:2]
    row_h = max(1, row["y2"] - row["y1"])
    rng = np.random.default_rng(seed)
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    metal_mask = np.zeros((h, w), dtype=np.uint8)
    light_mask = np.zeros((h, w), dtype=np.uint8)

    left = min(layout["outer"][0] for layout in layouts)
    right = max(layout["outer"][2] for layout in layouts)
    top = min(layout["outer"][1] for layout in layouts)
    bottom = max(layout["outer"][3] for layout in layouts)
    rail_pad = max(2, int(row_h * 0.18))
    rail_width = 1
    support_w = max(1, int(row_h * 0.045))
    foot_w = max(2, int(row_h * 0.14))
    foot_h = 1

    rail_ys = [
        int(top + row_h * 0.27),
        int(top + row_h * 0.73),
    ]
    x_start = clamp(left - rail_pad, 0, w - 1)
    x_end = clamp(right + rail_pad, x_start + 1, w - 1)

    style = {
        "grass_open": "ground",
        "agri_field_new": "ground",
        "desert_track": "ground",
        "floating_water": "floating",
        "rooftop_warehouse": "roof",
    }.get(location_id, "ground")

    for y in rail_ys:
        y = clamp(y, 0, h - 1)
        shadow_shift = max(1, row_h // 9)
        cv2.line(shadow_mask, (x_start + 1, y + shadow_shift), (x_end + 1, y + shadow_shift), 80, 1, cv2.LINE_AA)
        if style in {"roof", "floating"}:
            cv2.line(metal_mask, (x_start, y), (x_end, y), 95, rail_width, cv2.LINE_AA)
            cv2.line(light_mask, (x_start, max(0, y - 1)), (x_end, max(0, y - 1)), 70, 1, cv2.LINE_AA)

    boundaries = [layouts[0]["outer"][0]] + [layout["outer"][2] for layout in layouts]
    support_xs = []
    for idx, sx in enumerate(boundaries):
        if idx in (0, len(boundaries) - 1) or idx % 3 == 0:
            support_xs.append(sx)

    if style == "roof":
        for sx in support_xs:
            sx += int(rng.integers(-1, 2))
            pad_top = clamp(bottom + max(1, row_h // 16), 0, h - 1)
            pad_bottom = clamp(pad_top + foot_h, pad_top + 1, h - 1)
            fx1 = clamp(sx - foot_w, 0, w - 1)
            fx2 = clamp(sx + foot_w, fx1 + 1, w - 1)
            cv2.rectangle(shadow_mask, (fx1 + 1, pad_top + 1), (fx2 + 1, pad_bottom + 1), 65, -1)
            cv2.rectangle(metal_mask, (fx1, pad_top), (fx2, pad_bottom), 105, -1)
    elif style == "floating":
        for sx in support_xs:
            sx += int(rng.integers(-1, 2))
            marker_y = clamp(bottom + max(1, row_h // 18), 0, h - 1)
            cv2.circle(shadow_mask, (sx + 1, marker_y + 1), max(1, support_w + 1), 55, -1)
            cv2.circle(metal_mask, (sx, marker_y), max(1, support_w), 85, -1)
    else:
        # From nadir, ground-mount racks mostly read as contact shadows and
        # small dark post marks, not large visible legs.
        for sx in support_xs:
            marker_y = clamp(bottom + max(1, row_h // 20), 0, h - 1)
            cv2.circle(shadow_mask, (sx + 1, marker_y + 1), max(1, support_w + 1), 52, -1)

    shadow = cv2.GaussianBlur(shadow_mask, (0, 0), max(0.8, row_h * 0.13)).astype(np.float32) / 255.0
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - shadow[..., None] * 0.16), 0, 255).astype(np.uint8)

    metal = np.zeros((h, w, 3), dtype=np.uint8)
    metal[:, :] = (92, 100, 96)
    highlight = np.zeros_like(metal)
    highlight[:, :] = (150, 156, 150)
    metal_alpha = (metal_mask.astype(np.float32) / 255.0 * (0.28 if style != "roof" else 0.34))[..., None]
    light_alpha = (light_mask.astype(np.float32) / 255.0 * 0.18)[..., None]
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - metal_alpha) + metal.astype(np.float32) * metal_alpha, 0, 255).astype(np.uint8)
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - light_alpha) + highlight.astype(np.float32) * light_alpha, 0, 255).astype(np.uint8)
    return output_rgb


def composite_panel_row(output_rgb, row, module_count, seed, location_id=None, source=None, source_row=None):
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    x1, y1 = clamp(x1, 0, BASE_SIZE[0] - 1), clamp(y1, 0, BASE_SIZE[1] - 1)
    x2, y2 = clamp(x2, x1 + 1, BASE_SIZE[0]), clamp(y2, y1 + 1, BASE_SIZE[1])
    row_w, row_h = x2 - x1, y2 - y1
    source_textures = []
    if source is not None and source_row is not None:
        for module_number in range(1, module_count + 1):
            try:
                crop = source_module_crop(source, source_row, module_number, module_count).convert("RGB")
            except Exception:
                crop = None
            if crop is not None and crop.width >= 8 and crop.height >= 6:
                source_textures.append(np.array(crop))
    # Prefer actual photographed PV crops from the texture bank; use source-row crops only as fallback.
    rgb_patch_bank = []
    try:
        rgb_patch_bank = real_panel_textures(texture_group_for_location(location_id))
        if not rgb_patch_bank:
            rgb_patch_bank = actual_panel_model_textures()
    except Exception:
        rgb_patch_bank = []

    layouts = module_layouts_for_row(row, module_count)
    output_rgb = apply_mounting_hardware(output_rgb, row, module_count, seed_for(seed, "mounting-hardware"), location_id)
    terrain_image = Image.fromarray(output_rgb.copy()).convert("RGB")
    row_cells = cells_for_row(location_id, row["id"]) if location_id else []

    # Ground contact shadow: mostly below the near edge, so modules read as
    # tilted upward on racks instead of flat stickers.
    shadow = np.zeros(output_rgb.shape[:2], dtype=np.uint8)
    offset = max(1, int(row_h * 0.045))
    for layout in layouts:
        poly = module_visual_polygon(row, layout, "outer").copy()
        poly[:, 1] += offset
        poly[:, 0] = np.clip(poly[:, 0], 0, BASE_SIZE[0] - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, BASE_SIZE[1] - 1)
        cv2.fillPoly(shadow, [np.round(poly).astype(np.int32)], 180, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(0.8, row_h * 0.08)).astype(np.float32) / 255.0
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - shadow[..., None] * 0.10), 0, 255).astype(np.uint8)

    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        module_cells = [
            cell for cell in row_cells
            if cell.get("module_number") == layout["module_number"]
        ]
        source_patch = None
        if rgb_patch_bank:
            texture_idx = seed_for(seed, layout["module_number"], "photo-patch-choice") % len(rgb_patch_bank)
            source_patch = Image.fromarray(np.asarray(rgb_patch_bank[texture_idx]).astype(np.uint8)).convert("RGB")
        elif source_textures:
            source_patch = Image.fromarray(source_textures[(layout["module_number"] - 1) % len(source_textures)]).convert("RGB")
        patch = draw_realistic_module_patch(
            (max(4, ox2 - ox1), max(4, oy2 - oy1)),
            seed_for(seed, layout["module_number"], "module-patch"),
            terrain_image,
            (ox1, oy1, ox2, oy2),
            module_cells,
            source_patch=source_patch,
            location_id=location_id,
        )
        polygon = module_visual_polygon(row, layout, "outer")
        warped = warp_rgba_to_polygon(patch, polygon)
        output_rgb = alpha_composite_rgba_array(output_rgb, warped)
        inner_polygon = module_visual_polygon(row, layout, "inner")
        if source_patch is None:
            draw_projected_square_lattice(
                output_rgb,
                inner_polygon,
                ox1,
                oy1,
                ox2,
                oy2,
                seed_for(seed, layout["module_number"], "rgb-square-lattice"),
                thermal=False,
            )

        near_left = tuple(np.round(polygon[3]).astype(int))
        near_right = tuple(np.round(polygon[2]).astype(int))
        far_left = tuple(np.round(polygon[0]).astype(int))
        far_right = tuple(np.round(polygon[1]).astype(int))
        if source_patch is None:
            # Procedural fallback only: photo patches already contain their physical module frame.
            edge_w = 1
            drop = max(1, int(round((oy2 - oy1) * 0.012)))
            side_left = (near_left[0], min(BASE_SIZE[1] - 1, near_left[1] + drop))
            side_right = (near_right[0], min(BASE_SIZE[1] - 1, near_right[1] + drop))
            side_face = np.array([near_left, near_right, side_right, side_left], dtype=np.int32)
            cv2.fillPoly(output_rgb, [side_face], (30, 39, 48), cv2.LINE_AA)
            cv2.line(output_rgb, side_left, side_right, (18, 26, 34), 1, cv2.LINE_AA)
            cv2.line(output_rgb, near_left, near_right, (24, 34, 43), edge_w, cv2.LINE_AA)
            cv2.line(output_rgb, far_left, far_right, (70, 86, 97), 1, cv2.LINE_AA)
    return output_rgb


def prepare_agri_installation_ground(output_rgb, rows, seed):
    """Grade soft mounting corridors into the agri field before panels are drawn."""
    h, w = output_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    source = output_rgb.astype(np.float32)
    smoothed = cv2.GaussianBlur(source, (0, 0), 7.5)
    hsv = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2HSV)
    green_keep = ((hsv[:, :, 0] > 32) & (hsv[:, :, 0] < 96) & (hsv[:, :, 1] > 48)).astype(np.float32)
    green_keep = cv2.GaussianBlur(green_keep, (0, 0), 5.5)

    corridor = np.zeros((h, w), dtype=np.float32)
    service_tracks = np.zeros((h, w), dtype=np.float32)
    tram_tracks = np.zeros((h, w), dtype=np.float32)
    sorted_rows = sorted(rows, key=lambda item: item["y1"])
    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        pad_y = max(11, int(row_h * 1.05))
        pad_x = max(24, int(row_h * 1.6))
        x1 = clamp(row["x1"] - pad_x, 0, w - 1)
        x2 = clamp(row["x2"] + pad_x, x1 + 1, w)
        y1 = clamp(row["y1"] - pad_y, 0, h - 1)
        y2 = clamp(row["y2"] + pad_y, y1 + 1, h)
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(local, (x1, y1), (x2, y2), 255, -1)
        local = cv2.GaussianBlur(local, (0, 0), max(6.5, row_h * 0.42)).astype(np.float32) / 255.0
        rough_edge = cv2.GaussianBlur(rng.normal(1.0, 0.16, (h, w)).astype(np.float32), (0, 0), 5.0)
        corridor = np.maximum(corridor, np.clip(local * rough_edge, 0, 1))
        for ty in (row["y1"] - int(row_h * 0.42), row["y2"] + int(row_h * 0.42)):
            if 0 <= ty < h:
                track = np.zeros((h, w), dtype=np.uint8)
                cv2.line(track, (x1, ty), (x2, ty), 255, max(2, row_h // 5), cv2.LINE_AA)
                track = cv2.GaussianBlur(track, (0, 0), max(2.0, row_h * 0.18)).astype(np.float32) / 255.0
                service_tracks = np.maximum(service_tracks, np.clip(track * rough_edge, 0, 1))

    # Tractor tram lines: paired wheel marks in the working alleys between PV rows.
    x_min = max(10, min(row["x1"] for row in rows) - 18)
    x_max = min(w - 10, max(row["x2"] for row in rows) + 18)
    gaps = []
    for upper, lower in zip(sorted_rows, sorted_rows[1:]):
        gy1 = upper["y2"]
        gy2 = lower["y1"]
        if gy2 - gy1 >= 8:
            gaps.append((gy1, gy2))
    for gy1, gy2 in gaps:
        center_y = (gy1 + gy2) // 2
        lane_half = max(3, min(9, (gy2 - gy1) // 5))
        for offset in (-5, 5):
            y = clamp(center_y + offset, 0, h - 1)
            cv2.line(tram_tracks, (x_min, y), (x_max, y), 1.0, max(2, lane_half // 2), cv2.LINE_AA)
    tram_tracks = cv2.GaussianBlur(tram_tracks, (0, 0), 1.4)

    # Keep the lower green crop identity, while still allowing row beds to read.
    corridor *= 1.0 - np.clip(green_keep * 0.22, 0, 0.22)
    soil_tint = np.array([164, 136, 86], dtype=np.float32)
    soil_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.7)[..., None]
    graded = smoothed * 0.30 + soil_tint * 0.70 + soil_noise * np.array([8, 7, 5], dtype=np.float32)

    # Fine texture from the original prevents the pads from becoming fake flat boxes.
    high = np.clip(source - cv2.GaussianBlur(source, (0, 0), 9), -10, 10)
    graded = graded + high * 0.30
    mixed = source * (1 - corridor[..., None] * 0.92) + graded * (corridor[..., None] * 0.92)
    track_tint = np.array([112, 96, 68], dtype=np.float32)
    track_strength = service_tracks * (1 - np.clip(green_keep * 0.28, 0, 0.28))
    mixed = mixed * (1 - track_strength[..., None] * 0.48) + track_tint * (track_strength[..., None] * 0.48)
    tram_tint = np.array([84, 102, 55], dtype=np.float32)
    mixed = mixed * (1 - tram_tracks[..., None] * 0.34) + tram_tint * (tram_tracks[..., None] * 0.34)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def draw_site_railings(image_rgb, rows, location_id, seed):
    """Add subtle perimeter railings/fence posts around PV fields."""
    # Real nadir/aerial PV imagery rarely shows clean cartoon perimeter rails.
    # Site integration should come from row beds, service tracks, and contact
    # shadows, not an outlined rectangle around the array.
    return image_rgb
    if not rows or location_id in {"floating_water", "rooftop_warehouse"}:
        return image_rgb
    h, w = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    out = image_rgb.copy()
    x1 = max(6, min(row["x1"] for row in rows) - (32 if location_id in {"grass_open", "agri_field_new"} else 22))
    x2 = min(w - 7, max(row["x2"] for row in rows) + (32 if location_id in {"grass_open", "agri_field_new"} else 22))
    y1 = max(6, min(row["y1"] for row in rows) - (24 if location_id in {"grass_open", "agri_field_new"} else 18))
    y2 = min(h - 7, max(row["y2"] for row in rows) + (24 if location_id in {"grass_open", "agri_field_new"} else 18))

    rail_mask = np.zeros((h, w), dtype=np.uint8)
    post_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(rail_mask, (x1, y1), (x2, y2), 165, 2 if location_id in {"grass_open", "desert_track"} else 1)
    post_step = 26 if location_id in {"grass_open", "agri_field_new"} else 34
    for x in range(x1, x2 + 1, post_step):
        jitter = int(rng.integers(-1, 2))
        cv2.rectangle(post_mask, (clamp(x + jitter - 1, 0, w - 1), y1 - 1), (clamp(x + jitter + 1, 0, w - 1), y1 + 2), 160, -1)
        cv2.rectangle(post_mask, (clamp(x + jitter - 1, 0, w - 1), y2 - 2), (clamp(x + jitter + 1, 0, w - 1), y2 + 1), 145, -1)
    for y in range(y1, y2 + 1, post_step):
        jitter = int(rng.integers(-1, 2))
        cv2.rectangle(post_mask, (x1 - 1, clamp(y + jitter - 1, 0, h - 1)), (x1 + 2, clamp(y + jitter + 1, 0, h - 1)), 150, -1)
        cv2.rectangle(post_mask, (x2 - 2, clamp(y + jitter - 1, 0, h - 1)), (x2 + 1, clamp(y + jitter + 1, 0, h - 1)), 140, -1)

    shadow = cv2.GaussianBlur(np.maximum(rail_mask, post_mask), (0, 0), 0.7).astype(np.float32) / 255.0
    out = np.clip(out.astype(np.float32) * (1.0 - shadow[..., None] * 0.12), 0, 255).astype(np.uint8)
    rail_color = np.array([58, 68, 56] if location_id != "desert_track" else [68, 62, 50], dtype=np.float32)
    metal_alpha = ((rail_mask.astype(np.float32) / 255.0) * 0.50 + (post_mask.astype(np.float32) / 255.0) * 0.62)[..., None]
    out = np.clip(out.astype(np.float32) * (1 - metal_alpha) + rail_color * metal_alpha, 0, 255).astype(np.uint8)
    return out


def draw_mounting_rails(image_rgb, rows, location_id, seed):
    """Draw mounting rails and small support posts before the PV modules are composited."""
    if not rows:
        return image_rgb
    h, w = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    rail_mask = np.zeros((h, w), dtype=np.float32)
    post_mask = np.zeros((h, w), dtype=np.float32)
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    heavy_mounts = location_id in {"floating_water", "rooftop_warehouse"}

    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        x_extra = max(8 if heavy_mounts else 5, row_h // (2 if heavy_mounts else 3))
        x1 = clamp(row["x1"] - x_extra, 0, w - 1)
        x2 = clamp(row["x2"] + x_extra, x1 + 1, w - 1)
        rail_width = 2 if heavy_mounts else 1
        upper_y = clamp(row["y1"] - max(3 if heavy_mounts else 2, row_h // 8), 0, h - 1)
        lower_y = clamp(row["y2"] + max(3 if heavy_mounts else 2, row_h // 8), 0, h - 1)
        cv2.line(rail_mask, (x1, upper_y), (x2, upper_y), 1.0, rail_width, cv2.LINE_AA)
        cv2.line(rail_mask, (x1, lower_y), (x2, lower_y), 1.0, rail_width, cv2.LINE_AA)
        if heavy_mounts:
            mid_y = clamp((row["y1"] + row["y2"]) // 2, 0, h - 1)
            cv2.line(rail_mask, (x1, mid_y), (x2, mid_y), 0.72, 1, cv2.LINE_AA)

        module_count = max(1, custom_panels_per_row(location_id))
        step = max(18 if heavy_mounts else 24, int((row["x2"] - row["x1"]) / max(1, module_count)))
        for px in range(row["x1"] + step // 2, row["x2"], step):
            jitter = int(rng.integers(-2, 3))
            px = clamp(px + jitter, 0, w - 1)
            post_w = 2 if heavy_mounts else 1
            post_h = max(5 if heavy_mounts else 4, int(row_h * (0.42 if heavy_mounts else 0.28)))
            cv2.rectangle(post_mask, (px - post_w, lower_y), (px + post_w, clamp(lower_y + post_h, 0, h - 1)), 1.0, -1)
            cv2.rectangle(shadow_mask, (px - post_w + 2, lower_y + 2), (px + post_w + 5, clamp(lower_y + post_h + 4, 0, h - 1)), 1.0, -1)
        cv2.line(shadow_mask, (x1 + 2, lower_y + 2), (x2 + 3, lower_y + 2), 1.0, max(1, rail_width), cv2.LINE_AA)

    shadow = cv2.GaussianBlur(shadow_mask, (0, 0), 1.0 if heavy_mounts else 1.35)[..., None]
    rail = cv2.GaussianBlur(rail_mask, (0, 0), 0.45 if heavy_mounts else 0.55)[..., None]
    posts = cv2.GaussianBlur(post_mask, (0, 0), 0.55 if heavy_mounts else 0.70)[..., None]
    if location_id == "desert_track":
        rail_color = np.array([116, 103, 78], dtype=np.float32)
        post_color = np.array([100, 92, 72], dtype=np.float32)
    elif location_id == "floating_water":
        rail_color = np.array([94, 124, 130], dtype=np.float32)
        post_color = np.array([70, 94, 100], dtype=np.float32)
    elif location_id == "rooftop_warehouse":
        rail_color = np.array([116, 122, 122], dtype=np.float32)
        post_color = np.array([98, 102, 102], dtype=np.float32)
    else:
        rail_color = np.array([78, 88, 76], dtype=np.float32)
        post_color = np.array([68, 78, 70], dtype=np.float32)
    out = image_rgb.astype(np.float32)
    shadow_strength = 0.18 if heavy_mounts else 0.14
    rail_strength = 0.42 if heavy_mounts else 0.32
    post_strength = 0.46 if heavy_mounts else 0.34
    out *= 1.0 - shadow * shadow_strength
    out = out * (1.0 - rail * rail_strength) + rail_color * (rail * rail_strength)
    out = out * (1.0 - posts * post_strength) + post_color * (posts * post_strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare_floating_installation_environment(output_rgb, rows, seed):
    """Add visible pontoons, mooring rails, and service walkways for the offshore scene."""
    return output_rgb


def prepare_rooftop_installation_environment(output_rgb, rows, seed):
    """Add roof ballast pads and stronger metal rails under the rooftop array."""
    return output_rgb


def prepare_installation_environment(output_rgb, rows, location_id, seed):
    """Add site context behind custom PV rows: beds, service roads, and a small control pad."""
    if not rows:
        return output_rgb
    if location_id == "floating_water":
        return prepare_floating_installation_environment(output_rgb, rows, seed)
    if location_id == "rooftop_warehouse":
        return prepare_rooftop_installation_environment(output_rgb, rows, seed)
    if location_id == "agri_field_new":
        prepared = prepare_agri_installation_ground(output_rgb, rows, seed)
        prepared = draw_mounting_rails(prepared, rows, location_id, seed_for(seed, "mounts"))
        return draw_site_railings(prepared, rows, location_id, seed_for(seed, "railings"))

    h, w = output_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    source = output_rgb.astype(np.float32)
    mixed = source.copy()
    bed_mask = np.zeros((h, w), dtype=np.float32)
    road_mask = np.zeros((h, w), dtype=np.float32)
    asphalt_mask = np.zeros((h, w), dtype=np.float32)
    gravel_mask = np.zeros((h, w), dtype=np.float32)

    x1_all = max(0, min(row["x1"] for row in rows) - 18)
    x2_all = min(w - 1, max(row["x2"] for row in rows) + 18)
    y1_all = max(0, min(row["y1"] for row in rows) - 18)
    y2_all = min(h - 1, max(row["y2"] for row in rows) + 18)
    sorted_rows = sorted(rows, key=lambda item: item["y1"])
    row_gaps = []
    for upper, lower in zip(sorted_rows, sorted_rows[1:]):
        gy1 = upper["y2"]
        gy2 = lower["y1"]
        if gy2 - gy1 >= 8:
            row_gaps.append((gy1, gy2))

    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        pad_y = max(5, int(row_h * 0.85))
        pad_x = max(14, int(row_h * 1.15))
        bx1 = clamp(row["x1"] - pad_x, 0, w - 1)
        bx2 = clamp(row["x2"] + pad_x, bx1 + 1, w)
        by1 = clamp(row["y1"] - pad_y, 0, h - 1)
        by2 = clamp(row["y2"] + pad_y, by1 + 1, h)
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(local, (bx1, by1), (bx2, by2), 255, -1)
        local = cv2.GaussianBlur(local, (0, 0), max(2.2, row_h * 0.20)).astype(np.float32) / 255.0
        bed_mask = np.maximum(bed_mask, local)

    # Service tracks between rows. Avoid full perimeter boxes; in nadir imagery
    # those read as drawn outlines unless they already exist in the source photo.
    road_width = 7 if location_id == "grass_open" else (6 if location_id == "desert_track" else 5)
    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        y = clamp(row["y2"] + max(4, int(row_h * 0.8)), 0, h - 1)
        cv2.line(road_mask, (x1_all, y), (x2_all, y), 1.0, max(2, road_width // 2), cv2.LINE_AA)
        if location_id == "grass_open":
            y_top = clamp(row["y1"] - max(4, int(row_h * 0.75)), 0, h - 1)
            cv2.line(road_mask, (x1_all, y_top), (x2_all, y_top), 0.72, max(2, road_width // 3), cv2.LINE_AA)
    if location_id in {"grass_open", "desert_track"}:
        access_x = clamp(x1_all - 18, 0, w - 1)
        cv2.line(road_mask, (access_x, y1_all), (access_x, y2_all), 0.55, max(2, road_width // 2), cv2.LINE_AA)
        mid_y = clamp((y1_all + y2_all) // 2, 0, h - 1)
        cv2.line(road_mask, (access_x, mid_y), (x1_all, mid_y), 0.45, max(1, road_width // 3), cv2.LINE_AA)
    if location_id == "grass_open":
        for gy1, gy2 in row_gaps:
            lane_y = clamp((gy1 + gy2) // 2, 0, h - 1)
            lane_w = max(7, min(15, int((gy2 - gy1) * 0.36)))
            cv2.line(gravel_mask, (x1_all, lane_y), (x2_all, lane_y), 1.0, lane_w, cv2.LINE_AA)
    if location_id == "desert_track":
        for gy1, gy2 in row_gaps:
            lane_y = clamp((gy1 + gy2) // 2, 0, h - 1)
            lane_w = max(8, min(16, int((gy2 - gy1) * 0.34)))
            cv2.line(asphalt_mask, (x1_all, lane_y), (x2_all, lane_y), 1.0, lane_w, cv2.LINE_AA)
    road_mask = cv2.GaussianBlur(road_mask, (0, 0), 2.0)
    asphalt_mask = cv2.GaussianBlur(asphalt_mask, (0, 0), 1.2)
    gravel_mask = cv2.GaussianBlur(gravel_mask, (0, 0), 1.4)

    if location_id == "grass_open":
        bed_tint = np.array([112, 116, 80], dtype=np.float32)
        road_tint = np.array([136, 124, 88], dtype=np.float32)
        control_tint = np.array([160, 164, 150], dtype=np.float32)
        bed_alpha = 0.62
        road_alpha = 0.62
        bed_source = 0.56
    elif location_id == "desert_track":
        bed_tint = np.array([184, 163, 116], dtype=np.float32)
        road_tint = np.array([154, 132, 88], dtype=np.float32)
        control_tint = np.array([178, 170, 148], dtype=np.float32)
        bed_alpha = 0.46
        road_alpha = 0.58
        bed_source = 0.70
    else:
        bed_tint = np.array([117, 132, 82], dtype=np.float32)
        road_tint = np.array([132, 123, 92], dtype=np.float32)
        control_tint = np.array([178, 180, 165], dtype=np.float32)
        bed_alpha = 0.50
        road_alpha = 0.72
        bed_source = 0.58

    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.4)[..., None]
    beds = source * bed_source + bed_tint * (1.0 - bed_source) + noise * np.array([7, 6, 4], dtype=np.float32)
    mixed = mixed * (1 - bed_mask[..., None] * bed_alpha) + beds * (bed_mask[..., None] * bed_alpha)
    roads = source * 0.38 + road_tint * 0.62 + noise * np.array([5, 5, 4], dtype=np.float32)
    mixed = mixed * (1 - road_mask[..., None] * road_alpha) + roads * (road_mask[..., None] * road_alpha)
    if location_id == "grass_open":
        gravel_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.45)[..., None]
        gravel = source * 0.28 + np.array([158, 150, 126], dtype=np.float32) * 0.72 + gravel_noise * np.array([18, 17, 14], dtype=np.float32)
        mixed = mixed * (1 - gravel_mask[..., None] * 0.78) + gravel * (gravel_mask[..., None] * 0.78)
    if location_id == "desert_track":
        asphalt_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.65)[..., None]
        asphalt = source * 0.22 + np.array([76, 74, 68], dtype=np.float32) * 0.78 + asphalt_noise * np.array([10, 10, 9], dtype=np.float32)
        mixed = mixed * (1 - asphalt_mask[..., None] * 0.82) + asphalt * (asphalt_mask[..., None] * 0.82)

    # Control / maintenance pad: small, off to the side, connected by access road.
    pad_w = 34
    pad_h = 23
    if x1_all > 70:
        px1 = max(8, x1_all - pad_w - 18)
    else:
        px1 = min(w - pad_w - 8, x2_all + 18)
    py1 = clamp(y1_all + 5, 8, h - pad_h - 8)
    px2 = px1 + pad_w
    py2 = py1 + pad_h
    pad_mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(pad_mask, (px1, py1), (px2, py2), 1.0, -1)
    cv2.rectangle(pad_mask, (px1 - 4, py1 - 4), (px2 + 4, py2 + 4), 0.45, 1)
    pad_mask = cv2.GaussianBlur(pad_mask, (0, 0), 0.6)
    pad = source * 0.30 + control_tint * 0.70
    mixed = mixed * (1 - pad_mask[..., None] * 0.82) + pad * (pad_mask[..., None] * 0.82)

    # Simple roof/ac unit marks on the control pad.
    rect_color = (72, 78, 78) if location_id == "desert_track" else (64, 76, 66)
    mixed_u8 = np.clip(mixed, 0, 255).astype(np.uint8)
    cv2.rectangle(mixed_u8, (px1 + 5, py1 + 5), (px2 - 5, py2 - 5), rect_color, 1)
    cv2.rectangle(mixed_u8, (px1 + 9, py1 + 8), (px1 + 17, py1 + 15), rect_color, -1)
    cv2.line(mixed_u8, (px2, py1 + pad_h // 2), (x1_all, py1 + pad_h // 2), tuple(int(v) for v in road_tint), 2, cv2.LINE_AA)
    mixed_u8 = draw_mounting_rails(mixed_u8, rows, location_id, seed_for(seed, "mounts"))
    mixed_u8 = draw_site_railings(mixed_u8, rows, location_id, seed_for(seed, "railings"))
    return mixed_u8


def textured_alpha_mask(size, seed, feather=1.0):
    width, height = size
    mask = local_rect_mask(size, radius=0, feather=feather)
    if width < 4 or height < 4:
        return mask
    rng = random.Random(seed)
    edge_noise = Image.effect_noise(size, rng.uniform(8.0, 14.0)).convert("L")
    edge_noise = edge_noise.point(lambda value: int(232 + (value - 128) * 0.08))
    mask = ImageChops.multiply(mask, edge_noise)
    return mask.filter(ImageFilter.GaussianBlur(0.25))


def draw_realistic_module_patch(size, seed, terrain, box, module_cells, source_patch=None, location_id=None):
    width, height = size
    rng = random.Random(seed)
    width = max(4, int(width))
    height = max(4, int(height))
    terrain_crop = terrain.crop(box).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    terrain_level = mean_luma(terrain_crop)

    if source_patch is not None:
        # Preserve the photographed PV surface. Do not redraw a perfect grid or double-frame it.
        photo = source_patch.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        photo = match_patch_lighting(photo, terrain, box).convert("RGB")
        photo = ImageEnhance.Brightness(photo).enhance(rng.uniform(0.985, 1.025))
        photo = ImageEnhance.Contrast(photo).enhance(rng.uniform(0.985, 1.025))
        photo = harmonize_patch_texture(
            photo,
            terrain_crop,
            seed_for(seed, "photo-harmonize"),
            edge_strength=0.012,
        )
        arr = np.asarray(photo, dtype=np.float32)
        if location_id == "floating_water":
            # Subtle, broad water-color reflection; keep the captured cell texture dominant.
            env = np.asarray(terrain_crop, dtype=np.float32)
            env = cv2.GaussianBlur(env, (0, 0), max(2.0, min(width, height) * 0.16))
            arr = arr * 0.975 + env * 0.025
        grain = np.random.default_rng(seed_for(seed, "photo-sensor-grain")).normal(0, 0.45, arr.shape).astype(np.float32)
        photo = Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8), "RGB")
        result = photo.convert("RGBA")
        result.putalpha(local_rect_mask((width, height), radius=0, feather=0.28))
        return result

    base_r = rng.randint(18, 34)
    base_g = rng.randint(42, 62)
    base_b = rng.randint(70, 104)
    if source_patch is not None:
        real_rgb = procedural_panel_module((width, height), seed)
        if source_patch.size[0] >= 12 and source_patch.size[1] >= 6:
            source_texture = source_patch.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
            source_texture = ImageEnhance.Contrast(mute_bright_panel_lines(source_texture)).enhance(1.02)
            real_rgb = Image.blend(source_texture, real_rgb, 0.12)
        real_rgb = match_patch_lighting(real_rgb.convert("RGBA"), terrain, box).convert("RGB")
        tint = Image.new("RGB", (width, height), (base_r, base_g, base_b))
        real_rgb = Image.blend(real_rgb, tint, 0.025 if source_patch is not None else 0.08)
        module = real_rgb.convert("RGBA")
    else:
        module = Image.new("RGBA", (width, height), (base_r, base_g, base_b, 255))
    draw = ImageDraw.Draw(module)

    frame = max(1, min(width, height) // 13)
    inner = (frame, frame, width - frame - 1, height - frame - 1)
    frame_light = (142, 156, 164, 18)
    frame_dark = (15, 24, 31, 38)
    if source_patch is None:
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=max(1, frame), fill=frame_dark)
        draw.rectangle(inner, fill=(base_r, base_g, base_b, 255))
    else:
        pass

    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    inner_w = max(1, inner[2] - inner[0] + 1)
    inner_h = max(1, inner[3] - inner[1] + 1)
    gap = max(1, min(width, height) // 34)

    for cell in module_cells:
        cx = cell["module_cell_col"] - 1
        cy = cell["module_cell_row"] - 1
        x1 = int(inner[0] + cx * inner_w / cols) + gap
        y1 = int(inner[1] + cy * inner_h / rows) + gap
        x2 = int(inner[0] + (cx + 1) * inner_w / cols) - gap
        y2 = int(inner[1] + (cy + 1) * inner_h / rows) - gap
        if x2 <= x1 or y2 <= y1:
            continue
        radius = max(0, min(x2 - x1, y2 - y1) // 12)
        if source_patch is None:
            tint = rng.randint(-8, 9)
            cell_color = (
                clamp(base_r + tint, 6, 55),
                clamp(base_g + tint + rng.randint(-3, 4), 25, 82),
                clamp(base_b + tint + rng.randint(-5, 8), 52, 126),
                255,
            )
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=cell_color)
        else:
            # Preserve the real panel photograph texture; cell IDs are shown in
            # the zoom overlay, so the scene itself should not look annotated.
            pass
        # Subtle busbar lines inside each selectable cell.
        if source_patch is None and y2 - y1 >= 6:
            by = y1 + (y2 - y1) // 2
            draw.line((x1 + 1, by, x2 - 1, by), fill=(116, 132, 148, 6), width=1)
        if source_patch is None and x2 - x1 >= 8:
            bx = x1 + (x2 - x1) // 2
            draw.line((bx, y1 + 1, bx, y2 - 1), fill=(118, 132, 145, 5), width=1)

    # Real module seam/frame lines, stronger than cell busbars but not UI-white.
    seam = (185, 198, 202, 10 if source_patch is not None else 80)
    frame_seam = (205, 214, 216, 48 if source_patch is not None else 120)
    draw.rectangle((1, 1, width - 2, height - 2), outline=frame_seam, width=1)
    if source_patch is None:
        for idx in range(1, cols):
            x = int(inner[0] + idx * inner_w / cols)
            draw.line((x, inner[1], x, inner[3]), fill=seam, width=1)
        for idx in range(1, rows):
            y = int(inner[1] + idx * inner_h / rows)
            draw.line((inner[0], y, inner[2], y), fill=seam, width=1)

    # Glass glare and real imaging grain.
    glare = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glare_draw = ImageDraw.Draw(glare)
    glare_y = rng.randint(max(0, height // 8), max(1, height // 2))
    glare_draw.polygon(
        [
            (0, glare_y),
            (width, max(0, glare_y - height // 5)),
            (width, max(0, glare_y - height // 5 + max(2, height // 7))),
            (0, glare_y + max(2, height // 7)),
        ],
        fill=(255, 255, 255, rng.randint(0, 4) if source_patch is not None else rng.randint(12, 28)),
    )
    module.alpha_composite(glare.filter(ImageFilter.GaussianBlur(max(0.5, height * 0.018))))

    noise = Image.effect_noise((width, height), rng.uniform(6.5, 11.0)).convert("L")
    noise = noise.point(lambda value: int(128 + (value - 128) * (0.018 if source_patch is not None else 0.025)))
    rgb = ImageChops.multiply(module.convert("RGB"), noise.convert("RGB"))
    if source_patch is not None:
        rgb = ImageEnhance.Brightness(rgb).enhance(1.02)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.02)
        rgb = ImageEnhance.Color(rgb).enhance(1.00)
    # Small local exposure match only; large global boosts make glass look painted.
    brightness = clamp(0.94 + (terrain_level / 255.0) * 0.14, 0.96, 1.08)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.02 if source_patch is not None else 1.04)
    if source_patch is not None:
        rgb = harmonize_patch_texture(
            rgb,
            terrain_crop,
            seed_for(seed, "terrain-harmonize"),
            edge_strength=0.04,
        )
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.06)

    arr = np.array(rgb).astype(np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    tilt_light = 0.985 + yy * 0.030
    center_warmth = np.exp(-(xx ** 2 * 1.7 + (yy - 0.62) ** 2 * 4.2)) * (1.8 if source_patch is not None else 5.0)
    arr *= tilt_light[..., None]
    arr[:, :, 0] += center_warmth
    arr[:, :, 1] += center_warmth * 0.62
    arr[:, :, 2] -= center_warmth * 0.20
    rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    alpha = (
        local_rect_mask((width, height), radius=0, feather=0.35)
        if source_patch is not None
        else textured_alpha_mask((width, height), seed_for(seed, "alpha-texture"), feather=0.45)
    )
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def draw_module_detail(draw, box, seed, selected_units, cell=None):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    rng = random.Random(seed)

    minor = (226, 236, 244, 56)
    major = (232, 240, 246, 132)
    shadow = (18, 32, 55, 46)

    # Real PV-cell boundary: faint on every cell so the fault can be checked by
    # counting cells, with a heavier divider at module edges.
    draw.rounded_rectangle((x1, y1, x2, y2), radius=max(1, min(width, height) // 7), outline=minor, width=1)
    if height >= 8:
        bus1 = int(y1 + height * 0.36)
        bus2 = int(y1 + height * 0.67)
        draw.line((x1 + 1, bus1, x2 - 1, bus1), fill=(220, 230, 244, 36), width=1)
        draw.line((x1 + 1, bus2, x2 - 1, bus2), fill=(220, 230, 244, 30), width=1)

    if cell and cell.get("module_cell_col") == 1:
        draw.line((x1, y1, x1, y2), fill=major, width=2)
    if cell and cell.get("module_cell_col") == CUSTOM_MODULE_CELL_COLS:
        draw.line((x2, y1, x2, y2), fill=shadow, width=2)
    if cell and cell.get("module_cell_row") == 1:
        draw.line((x1, y1, x2, y1), fill=major, width=1)
    if cell and cell.get("module_cell_row") == CUSTOM_MODULE_CELL_ROWS:
        draw.line((x1, y2, x2, y2), fill=shadow, width=1)


def custom_panel_scene(location_id):
    """Build real visible PV rows/cells for the current custom layout."""
    location = location_by_id(location_id)
    terrain = terrain_layer(location_id).convert("RGBA")
    if location_id == "desert_track":
        source = load_rgb("desert_user_reference_with_pvs.png").convert("RGB")
        source_rows = DESERT_USER_ROW_DEFS
    elif location.get("pv_source") and location.get("pv_source") != location.get("source"):
        source = load_rgb(location["pv_source"]).convert("RGB")
        source_rows = ROW_DEFS if location["pv_source"] == "scenario_1_og_pv.jpeg" else base_row_defs_for_location(location_id)
    else:
        # A panel-free background is not a panel texture: use the real existing PV reference instead.
        source = load_rgb("scenario_1_og_pv.jpeg").convert("RGB")
        source_rows = ROW_DEFS
    output_rgb = np.array(terrain.convert("RGB"), dtype=np.uint8)
    module_count = custom_panels_per_row(location_id)
    rows = panels(location_id)
    output_rgb = prepare_installation_environment(
        output_rgb,
        rows,
        location_id,
        seed_for(location_id, module_count, "installation-environment"),
    )
    source_rows = sorted(source_rows, key=lambda item: (item["y1"] + item["y2"]) / 2)
    for row_index, row in enumerate(rows):
        source_row = source_rows[min(row_index, len(source_rows) - 1)] if source_rows else None
        output_rgb = composite_panel_row(
            output_rgb,
            row,
            module_count,
            seed_for(location_id, row["id"], module_count, "row-level-composite"),
            location_id,
            source=source,
            source_row=source_row,
        )

    return Image.fromarray(output_rgb)


def composite_with_og_pvs(location_id):
    location = location_by_id(location_id)
    if custom_layout_enabled():
        return custom_panel_scene(location_id)
    if location.get("pv_source"):
        return load_rgb(location["pv_source"])

    image = terrain_layer(location_id).convert("RGBA")
    image.alpha_composite(pv_cutout())
    return image.convert("RGB")
def seed_for(*parts):
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def irregular_mask(size, seed, blobs=7):
    width, height = size
    rng = random.Random(seed)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(blobs):
        cx = rng.randint(max(1, width // 5), max(2, width * 4 // 5))
        cy = rng.randint(max(1, height // 5), max(2, height * 4 // 5))
        rx = rng.randint(max(2, width // 8), max(3, width // 3))
        ry = rng.randint(max(2, height // 8), max(3, height // 2))
        fill = rng.randint(110, 230)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)
    return ImageOps.autocontrast(mask.filter(ImageFilter.GaussianBlur(max(1.0, min(width, height) * 0.06))))


def paste_color_with_mask(image, color, box, mask, alpha=255):
    patch = Image.new("RGBA", mask.size, (*color, alpha))
    patch.putalpha(mask.point(lambda value: int(value * alpha / 255)))
    image.alpha_composite(patch, box[:2])


def pil_hotspot_mask(size, seed, severity):
    width, height = size
    rng = random.Random(seed)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    cx = width * (0.42 + rng.uniform(-0.08, 0.10))
    cy = height * (0.50 + rng.uniform(-0.10, 0.10))
    rx = max(2, int(width * (0.17 + 0.12 * severity)))
    ry = max(2, int(height * (0.20 + 0.13 * severity)))
    for idx, alpha in enumerate((80, 135, 210)):
        grow = 2 - idx
        jitter_x = rng.randint(-max(1, rx // 7), max(1, rx // 7))
        jitter_y = rng.randint(-max(1, ry // 7), max(1, ry // 7))
        draw.ellipse(
            (
                int(cx - rx * (1 + grow * 0.55) + jitter_x),
                int(cy - ry * (1 + grow * 0.55) + jitter_y),
                int(cx + rx * (1 + grow * 0.55) + jitter_x),
                int(cy + ry * (1 + grow * 0.55) + jitter_y),
            ),
            fill=alpha,
        )
    noise = Image.effect_noise(size, rng.uniform(9.0, 15.0)).convert("L")
    noise = noise.point(lambda value: int(clamp(202 + (value - 128) * 0.26, 0, 255)))
    return ImageChops.multiply(mask.filter(ImageFilter.GaussianBlur(max(0.6, min(width, height) * 0.055))), noise)


def draw_realistic_fault(image, cell, fault_type, scale, view):
    fault_type = canonical_fault_type(fault_type)
    image = image.convert("RGBA")
    x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = max(2, int(w * 0.08))
    pad_y = max(1, int(h * 0.12))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    if bx2 <= bx1 or by2 <= by1:
        bx1, by1, bx2, by2 = x1, y1, x2, y2
    if bx2 - bx1 < 4:
        cx = (bx1 + bx2) // 2
        bx1 = clamp(cx - 2, 0, BASE_SIZE[0] - 4)
        bx2 = bx1 + 4
    if by2 - by1 < 4:
        cy = (by1 + by2) // 2
        by1 = clamp(cy - 2, 0, BASE_SIZE[1] - 4)
        by2 = by1 + 4
    bw, bh = bx2 - bx1, by2 - by1
    rng = random.Random(seed_for(cell["id"], fault_type, scale, view))
    severity = scale / 10
    overlay = Image.new("RGBA", BASE_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if fault_type in HOTSPOT_FAULTS:
        spot_pad = max(2, int(max(bw, bh) * 0.42))
        spot_count = 1 if fault_type == "SingleHotSpot" else rng.randint(2, 4)
        for _ in range(spot_count):
            cx = int(bx1 + bw * rng.uniform(0.28, 0.72))
            cy = int(by1 + bh * rng.uniform(0.30, 0.70))
            sx1 = clamp(cx - spot_pad, bx1, bx2 - 1)
            sy1 = clamp(cy - spot_pad, by1, by2 - 1)
            sx2 = clamp(cx + spot_pad, sx1 + 1, bx2)
            sy2 = clamp(cy + spot_pad, sy1 + 1, by2)
            sw, sh = sx2 - sx1, sy2 - sy1
            hot_mask = pil_hotspot_mask((sw, sh), rng.randint(0, 10_000), severity)
            core_mask = hot_mask.point(lambda value: 255 if value > 176 else 0).filter(ImageFilter.GaussianBlur(0.8))
            if view == "thermal":
                paste_color_with_mask(overlay, (244, 136, 26), (sx1, sy1, sx2, sy2), hot_mask, int(120 + 55 * severity))
                paste_color_with_mask(overlay, (255, 246, 126), (sx1, sy1, sx2, sy2), core_mask, int(130 + 85 * severity))
            else:
                paste_color_with_mask(overlay, (115, 54, 28), (sx1, sy1, sx2, sy2), hot_mask, int(55 + 45 * severity))

    elif fault_type in DIODE_FAULTS:
        band_count = 1 if fault_type in {"SingleDiode", "DB"} else 2
        for band in range(band_count):
            offset = int((band - (band_count - 1) / 2) * bh * 0.34)
            cy = int((by1 + by2) / 2 + offset)
            band_h = max(3, int(bh * (0.28 + severity * 0.18)))
            jitter = rng.randint(-max(1, bw // 20), max(1, bw // 20))
            local_w = max(4, bw)
            local_h = max(4, band_h * 2)
            band_mask = Image.new("L", (local_w, local_h), 0)
            band_draw = ImageDraw.Draw(band_mask)
            mid = local_h // 2
            rough = max(1, band_h // 5)
            points = [
                (0, mid - band_h // 2 + rng.randint(-rough, rough)),
                (local_w - 1, mid - band_h // 2 + rng.randint(-rough, rough)),
                (local_w - 1, mid + band_h // 2 + rng.randint(-rough, rough)),
                (0, mid + band_h // 2 + rng.randint(-rough, rough)),
            ]
            band_draw.polygon(points, fill=220)
            band_mask = ImageChops.multiply(
                band_mask.filter(ImageFilter.GaussianBlur(max(0.5, band_h * 0.08))),
                Image.effect_noise((local_w, local_h), rng.uniform(7.0, 12.0)).convert("L").point(lambda value: int(220 + (value - 128) * 0.12)),
            )
            box_x1 = clamp(bx1 + jitter, bx1, bx2 - local_w)
            box_y1 = clamp(cy - local_h // 2, by1, by2 - local_h)
            box = (box_x1, box_y1, box_x1 + local_w, box_y1 + local_h)
            if view == "thermal":
                paste_color_with_mask(overlay, (240, 172, 45), box, band_mask, int(125 + 70 * severity))
                core = band_mask.point(lambda value: 255 if value > 170 else 0).filter(ImageFilter.GaussianBlur(max(0.5, band_h * 0.06)))
                paste_color_with_mask(overlay, (255, 244, 150), box, core, int(80 + 55 * severity))
            else:
                paste_color_with_mask(overlay, (62, 71, 86), box, band_mask, int(65 + 35 * severity))

    elif fault_type in BYPASSED_FAULTS:
        section_count = 1 if fault_type == "SingleByPassed" else 2
        section_w = max(4, int(bw * 0.34))
        for idx in range(section_count):
            sx1 = clamp(int(bx1 + (idx + 0.5) * bw / section_count - section_w / 2), bx1, bx2 - section_w)
            sx2 = sx1 + section_w
            mask = irregular_mask((section_w, bh), rng.randint(0, 10_000), blobs=5)
            if view == "thermal":
                paste_color_with_mask(overlay, (242, 154, 38), (sx1, by1, sx2, by2), mask, int(70 + 70 * severity))
            else:
                paste_color_with_mask(overlay, (72, 76, 76), (sx1, by1, sx2, by2), mask, int(45 + 35 * severity))

    elif fault_type in CRACKING_FAULTS:
        patch = selected_fault_patch("CellCracking", seed_for(cell["id"], scale, view, "rgb-crack-patch"))
        mask_array = None
        if patch is not None:
            _, patch_mask = crack_patch_texture_mask(patch, (bw, bh), seed_for(cell["id"], scale, view, "rgb-crack-field"))
            if patch_mask is not None:
                edge = cv2.GaussianBlur(
                    np.abs(cv2.Laplacian(patch_mask.astype(np.float32), cv2.CV_32F)),
                    (0, 0),
                    max(0.30, min(bw, bh) * 0.012),
                )
                mask_array = np.clip(patch_mask * 0.86 + edge * 0.42, 0, 1)
        if mask_array is None:
            fallback = organic_fault_mask(bw, bh, "CellCracking", seed_for(cell["id"], scale, view, "rgb-crack-fallback"))
            mask_array = np.clip((fallback - np.percentile(fallback, 76)) / 0.24, 0, 1) * cell_edge_feather(bw, bh, strength=9.0)
        preserve = panel_segmentation_preserve_mask((BASE_SIZE[1], BASE_SIZE[0], 3), cell)[by1:by2, bx1:bx2]
        mask_array = np.clip(mask_array * (1.0 - preserve * 0.48), 0, 1)
        soft_mask = Image.fromarray(np.uint8(np.clip(mask_array, 0, 1) * 255), mode="L")
        soft_mask = soft_mask.filter(ImageFilter.GaussianBlur(max(0.25, min(bw, bh) * 0.010)))
        if view == "thermal":
            paste_color_with_mask(overlay, (248, 142, 42), (bx1, by1, bx2, by2), soft_mask, int(112 + 74 * severity))
            core = soft_mask.point(lambda value: 255 if value > 150 else 0).filter(ImageFilter.GaussianBlur(0.7))
            paste_color_with_mask(overlay, (255, 232, 130), (bx1, by1, bx2, by2), core, int(70 + 70 * severity))
        else:
            shadow = soft_mask.filter(ImageFilter.GaussianBlur(max(0.4, min(bw, bh) * 0.030)))
            paste_color_with_mask(overlay, (38, 45, 56), (bx1, by1, bx2, by2), shadow, int(36 + 32 * severity))
            paste_color_with_mask(overlay, (230, 234, 220), (bx1, by1, bx2, by2), soft_mask, int(126 + 74 * severity))

    elif fault_type in SOILING_FAULTS | {"Bird droppings", "Snow cover"}:
        mask = None
        if fault_type in SOILING_FAULTS:
            patch = selected_fault_patch("SoilingDust", seed_for(cell["id"], scale, view, "rgb-soil-patch"))
            if patch is not None:
                field, patch_alpha = normalized_patch_field(patch, (bw, bh), seed_for(cell["id"], scale, view, "rgb-soil-field"))
                if patch_alpha is not None:
                    soil_mask = np.clip(np.maximum(patch_alpha * 0.95, organic_fault_mask(bw, bh, fault_type, seed_for(cell["id"], scale, view, "rgb-soil-mask")) * 0.52), 0, 1)
                    mask = Image.fromarray(np.uint8(soil_mask * 255), mode="L").filter(ImageFilter.GaussianBlur(max(0.35, min(bw, bh) * 0.015)))
        if mask is None:
            mask = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=9)
        if view == "thermal":
            if fault_type == "Snow cover":
                color = (18, 26, 84)
                alpha = int(135 + 78 * severity)
            elif fault_type == "Bird droppings":
                color = (28, 24, 62)
                alpha = int(122 + 72 * severity)
            elif fault_type in SOILING_FAULTS:
                color = (34, 20, 92)
                alpha = int(154 + 62 * severity)
            else:
                color = (22, 38, 98)
                alpha = int(105 + 70 * severity)
            paste_color_with_mask(overlay, color, (bx1, by1, bx2, by2), mask, alpha)
            edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.8))
            paste_color_with_mask(overlay, (79, 184, 141), (bx1, by1, bx2, by2), edge, 80)
        else:
            if fault_type == "Snow cover":
                paste_color_with_mask(overlay, (224, 232, 232), (bx1, by1, bx2, by2), mask, int(150 + 70 * severity))
                edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.6))
                paste_color_with_mask(overlay, (170, 188, 198), (bx1, by1, bx2, by2), edge, 100)
            elif fault_type == "Bird droppings":
                paste_color_with_mask(overlay, (210, 204, 178), (bx1, by1, bx2, by2), mask, int(145 + 60 * severity))
                speck = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=4).point(lambda value: 255 if value > 168 else 0)
                paste_color_with_mask(overlay, (70, 64, 48), (bx1, by1, bx2, by2), speck, int(75 + 40 * severity))
            else:
                paste_color_with_mask(overlay, (104, 84, 62), (bx1, by1, bx2, by2), mask, int(140 + 54 * severity))

    elif fault_type in SURFACE_OBSTRUCTION_FAULTS:
        points = [
            (bx1 - int(bw * 0.2), by1 + int(bh * 0.15)),
            (bx2 + int(bw * 0.15), by1 + int(bh * 0.45)),
            (bx2 + int(bw * 0.15), by2 + int(bh * 0.15)),
            (bx1 - int(bw * 0.2), by2 - int(bh * 0.15)),
        ]
        fill = (18, 30, 80, int(120 + 65 * severity)) if view == "thermal" else (18, 24, 32, int(80 + 55 * severity))
        draw.polygon(points, fill=fill)
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.6, bh * 0.035)))

    elif fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS | THERMAL_BLOCK_FAULTS:
        if fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
            band_alpha = int(105 + 68 * severity)
            if view == "thermal":
                if fault_type == "StringReversedPolarity":
                    draw.line((bx1, by1 + int(bh * 0.32), bx2, by2 - int(bh * 0.22)), fill=(255, 238, 132, band_alpha), width=max(1, bw // 5))
                    draw.line((bx1, by2 - int(bh * 0.26), bx2, by1 + int(bh * 0.25)), fill=(232, 104, 40, int(band_alpha * 0.72)), width=max(1, bw // 7))
                else:
                    draw.line((bx1 + bw // 2, by1, bx1 + bw // 2, by2), fill=(255, 238, 132, band_alpha), width=max(1, bw // 5))
            else:
                draw.line((bx1 + bw // 2, by1, bx1 + bw // 2, by2), fill=(42, 48, 52, int(65 + 40 * severity)), width=max(1, bw // 5))
        else:
            mask = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=5)
            if view == "thermal":
                paste_color_with_mask(overlay, (248, 162, 48), (bx1, by1, bx2, by2), mask, int(80 + 70 * severity))
            else:
                paste_color_with_mask(overlay, (80, 72, 62), (bx1, by1, bx2, by2), mask, int(42 + 38 * severity))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.35, min(bw, bh) * 0.03)))

    else:
        points = []
        steps = 6
        for idx in range(steps + 1):
            t = idx / steps
            px = int(bx1 + bw * t)
            py = int(by1 + bh * (0.25 + 0.55 * t) + rng.randint(-max(1, bh // 5), max(1, bh // 5)))
            points.append((px, py))
        line_color = (12, 16, 34, int(145 + 70 * severity)) if view == "thermal" else (238, 242, 246, int(120 + 60 * severity))
        draw.line(points, fill=line_color, width=max(1, int(1 + severity * 3)))
        if view == "thermal":
            for px, py in points[1:-1:2]:
                r = max(1, int(min(bw, bh) * (0.07 + 0.04 * severity)))
                draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 230, 118, int(110 + 90 * severity)))

    image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(0.25)))
    return image.convert("RGB")
