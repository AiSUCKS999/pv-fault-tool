# Thermal conversion, thermal fault rendering, and thermal color tints.
def thermal_palette(value):
    stops = [
        (0, (18, 7, 32)),
        (42, (45, 14, 73)),
        (82, (88, 24, 101)),
        (118, (141, 39, 91)),
        (150, (190, 62, 61)),
        (184, (225, 105, 35)),
        (220, (246, 168, 48)),
        (255, (255, 225, 128)),
    ]
    for idx in range(len(stops) - 1):
        left_v, left_c = stops[idx]
        right_v, right_c = stops[idx + 1]
        if value <= right_v:
            ratio = 0 if right_v == left_v else (value - left_v) / (right_v - left_v)
            return tuple(int(left_c[channel] + (right_c[channel] - left_c[channel]) * ratio) for channel in range(3))
    return stops[-1][1]


@st.cache_data(show_spinner=False)
def thermal_lut():
    return [thermal_palette(value) for value in range(256)]


def temperature_to_rgb(temp):
    palette = []
    for color in thermal_lut():
        palette.extend(color)
    pal_img = temp.convert("P")
    pal_img.putpalette(palette)
    return pal_img.convert("RGB")


def thermal_terrain_settings(location_id):
    settings = {
        "scenario_1": (146, 42),
        "agri_rows": (150, 44),
        "desert_farm": (108, 40),
        "farm_lake": (118, 34),
        "rooftop": (150, 38),
    }
    return settings.get(location_id, (146, 40))


def thermal_overlay_alpha(location_id):
    settings = {
        "desert_farm": 0.60,
        "scenario_1": 0.66,
        "agri_rows": 0.62,
        "farm_lake": 0.58,
        "rooftop": 0.66,
    }
    return settings.get(location_id, 0.64)


def thermal_noise(shape, rng, amplitude, sigma):
    noise = rng.normal(0, amplitude, size=shape).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), sigma)


def blend_temp_patch(temp, box, patch, alpha):
    x1, y1, x2, y2 = box
    h, w = temp.shape
    x1, x2 = clamp(x1, 0, w), clamp(x2, 0, w)
    y1, y2 = clamp(y1, 0, h), clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return
    local = patch[: y2 - y1, : x2 - x1]
    temp[y1:y2, x1:x2] = temp[y1:y2, x1:x2] * (1 - alpha) + local * alpha


def real_fault_patch_boxes(fault_type, bx1, by1, bx2, by2, rng):
    bw, bh = max(3, bx2 - bx1), max(3, by2 - by1)
    if fault_type in HOTSPOT_FAULTS:
        count = 1 if fault_type == "SingleHotSpot" else int(rng.integers(2, 5))
        boxes = []
        for _ in range(count):
            if fault_type == "SingleHotSpot":
                sw = max(3, int(bw * rng.uniform(0.34, 0.56)))
                sh = max(3, int(bh * rng.uniform(0.40, 0.66)))
            else:
                sw = max(3, int(bw * rng.uniform(0.08, 0.15)))
                sh = max(3, int(bh * rng.uniform(0.28, 0.48)))
            cx = int(bx1 + bw * rng.uniform(0.24, 0.76))
            cy = int(by1 + bh * rng.uniform(0.24, 0.76))
            x1 = clamp(cx - sw // 2, bx1, bx2 - sw)
            y1 = clamp(cy - sh // 2, by1, by2 - sh)
            boxes.append((x1, y1, x1 + sw, y1 + sh))
        return boxes
    if fault_type in CRACKING_FAULTS:
        sw = max(5, int(bw * rng.uniform(0.82, 1.00)))
        sh = max(5, int(bh * rng.uniform(0.82, 1.00)))
        cx = int(bx1 + bw * rng.uniform(0.45, 0.55))
        cy = int(by1 + bh * rng.uniform(0.45, 0.55))
        x1 = clamp(cx - sw // 2, bx1, bx2 - sw)
        y1 = clamp(cy - sh // 2, by1, by2 - sh)
        return [(x1, y1, x1 + sw, y1 + sh)]
    if fault_type in DIODE_FAULTS:
        count = 1 if fault_type in {"SingleDiode", "DB"} else 2
        band_h = max(3, int(bh * (0.56 if count == 1 else 0.42)))
        boxes = []
        for idx in range(count):
            cy = int(by1 + (idx + 0.5) * bh / count + rng.normal(0, bh * 0.05))
            y1 = clamp(cy - band_h // 2, by1, by2 - band_h)
            inset = int(bw * rng.uniform(0.00, 0.08))
            boxes.append((bx1 + inset, y1, bx2 - inset, y1 + band_h))
        return boxes
    if fault_type in BYPASSED_FAULTS:
        count = 1 if fault_type == "SingleByPassed" else 2
        section_w = max(3, int(bw * (0.62 if count == 1 else 0.42)))
        boxes = []
        for idx in range(count):
            cx = int(bx1 + (idx + 0.5) * bw / count + rng.normal(0, bw * 0.04))
            x1 = clamp(cx - section_w // 2, bx1, bx2 - section_w)
            y_pad = int(bh * rng.uniform(0.00, 0.08))
            boxes.append((x1, by1 + y_pad, x1 + section_w, by2 - y_pad))
        return boxes
    if fault_type in SURFACE_OBSTRUCTION_FAULTS | SOILING_FAULTS:
        inset_x = int(bw * rng.uniform(0.00, 0.08))
        inset_y = int(bh * rng.uniform(0.00, 0.10))
        return [(bx1 + inset_x, by1 + inset_y, bx2 - inset_x, by2 - inset_y)]
    if fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
        stripe_w = max(3, int(bw * (0.55 if fault_type == "StringOpenCircuit" else 0.78)))
        cx = int(bx1 + bw * rng.uniform(0.38, 0.62))
        x1 = clamp(cx - stripe_w // 2, bx1, bx2 - stripe_w)
        return [(x1, by1, x1 + stripe_w, by2)]
    return [(bx1, by1, bx2, by2)]


def organic_fault_mask(width, height, fault_type, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if fault_type in HOTSPOT_FAULTS:
        cx = width * rng.uniform(0.44, 0.56)
        cy = height * rng.uniform(0.42, 0.58)
        rx = max(1.0, width * rng.uniform(0.18, 0.30))
        ry = max(1.0, height * rng.uniform(0.20, 0.34))
        base = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * rng.uniform(1.35, 1.85))
        rough = smooth_unit_field((height, width), seed_for(seed, "hotspot-organic"), max(0.8, min(width, height) * 0.13), octaves=3)
        mask = base * (0.72 + rough * 0.38)
    elif fault_type in DIODE_FAULTS:
        center = height * rng.uniform(0.42, 0.58)
        band = max(1.0, height * rng.uniform(0.30, 0.43))
        mask = np.exp(-((yy - center) / band) ** 2)
        mask *= 0.88 + 0.34 * smooth_unit_field((height, width), seed_for(seed, "diode-organic"), max(1.0, width * 0.16), octaves=2)
    elif fault_type in BYPASSED_FAULTS:
        mask = 0.82 + 0.30 * smooth_unit_field((height, width), seed_for(seed, "bypass-organic"), max(1.0, min(width, height) * 0.18), octaves=2)
        edge = np.minimum.reduce([xx / max(1, width - 1), 1 - xx / max(1, width - 1), yy / max(1, height - 1), 1 - yy / max(1, height - 1)])
        mask *= np.clip(edge * 6.0, 0, 1)
    elif fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
        if fault_type == "StringReversedPolarity":
            diag = np.abs((yy / max(1, height - 1)) - (xx / max(1, width - 1)))
            anti = np.abs((yy / max(1, height - 1)) - (1 - xx / max(1, width - 1)))
            mask = np.maximum(np.exp(-(diag / 0.25) ** 2), 0.70 * np.exp(-(anti / 0.32) ** 2))
        else:
            center = width * rng.uniform(0.45, 0.55)
            band = max(1.0, width * rng.uniform(0.34, 0.48))
            mask = np.exp(-((xx - center) / band) ** 2)
        mask *= 0.86 + 0.28 * smooth_unit_field((height, width), seed_for(seed, "string-organic"), max(1.0, min(width, height) * 0.20), octaves=2)
    elif fault_type in SURFACE_OBSTRUCTION_FAULTS:
        slope = rng.uniform(-0.32, 0.32)
        intercept = height * rng.uniform(0.25, 0.45)
        boundary = intercept + slope * (xx - width * 0.5)
        soft = np.clip((yy - boundary) / max(1.0, height * 0.22), 0, 1)
        rough = smooth_unit_field((height, width), seed_for(seed, "shade-organic"), max(1.0, min(width, height) * 0.22), octaves=3)
        mask = np.clip(soft * (0.72 + 0.42 * rough), 0, 1)
    elif fault_type in SOILING_FAULTS:
        broad = smooth_unit_field((height, width), seed_for(seed, "soil-broad"), max(1.0, min(width, height) * 0.24), octaves=4)
        speckle = smooth_unit_field((height, width), seed_for(seed, "soil-speckle"), max(0.7, min(width, height) * 0.055), octaves=3)
        clumps = np.clip((broad - 0.38) / 0.38, 0, 1)
        dust = np.clip((speckle - 0.42) / 0.34, 0, 1)
        mask = np.clip(clumps * 0.78 + dust * 0.34, 0, 1)
        mask *= cell_edge_feather(width, height, strength=6.0)
    elif fault_type in CRACKING_FAULTS:
        cloudy = smooth_unit_field((height, width), seed_for(seed, "crack-cloud"), max(1.0, min(width, height) * 0.20), octaves=4)
        fine = smooth_unit_field((height, width), seed_for(seed, "crack-fine"), max(0.7, min(width, height) * 0.045), octaves=3)
        edge = np.abs(cv2.Laplacian(fine.astype(np.float32), cv2.CV_32F))
        impact = np.exp(-(((xx - width * 0.52) / max(1.0, width * 0.34)) ** 2 + ((yy - height * 0.50) / max(1.0, height * 0.40)) ** 2))
        mask = np.clip((cloudy - 0.36) / 0.42, 0, 1) * impact
        mask = np.maximum(mask, np.clip(edge * 1.8, 0, 1) * impact)
    else:
        mask = smooth_unit_field((height, width), seed_for(seed, "block-organic"), max(1.0, min(width, height) * 0.20), octaves=2)

    coarse = smooth_unit_field((height, width), seed_for(seed, "fault-coarse-breakup"), max(1.0, min(width, height) * 0.11), octaves=3)
    fine = smooth_unit_field((height, width), seed_for(seed, "fault-fine-islands"), max(0.7, min(width, height) * 0.045), octaves=2)
    breakup = np.clip((coarse - 0.16) / 0.72, 0.18, 1.0)
    islands = np.clip((fine - 0.50) / 0.28, 0, 1)
    mask = mask * (0.70 + coarse * 0.42) * breakup
    mask = np.maximum(mask, mask * islands * 1.16)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.45, min(width, height) * 0.045))
    return np.clip(mask, 0, 1)


def normalized_patch_field(patch, size, seed):
    width, height = size
    if patch is None or width <= 1 or height <= 1:
        return None, None
    rgb = patch[:, :, :3] if patch.ndim == 3 and patch.shape[2] >= 3 else patch
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    low, high = np.percentile(gray, [8, 96])
    if high - low < 4:
        return None, None
    field = np.clip((gray - low) / (high - low), 0, 1)
    field = cv2.GaussianBlur(field, (0, 0), max(0.45, min(width, height) * 0.045))
    texture = smooth_unit_field((height, width), seed, max(1.0, min(width, height) * 0.18), octaves=2)
    field = np.clip(field * 0.82 + texture * 0.18, 0, 1)
    alpha = np.clip((field - 0.18) / 0.64, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.45, min(width, height) * 0.055))
    if patch.ndim == 3 and patch.shape[2] == 4:
        mask = cv2.resize(patch[:, :, 3].astype(np.float32) / 255.0, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), max(0.35, min(width, height) * 0.025))
        alpha = np.clip(np.maximum(alpha * 0.38, mask), 0, 1)
        field = np.clip(field * (0.46 + mask * 0.74), 0, 1)
    return field, alpha


def cell_edge_feather(width, height, strength=8.0):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    edge = np.minimum.reduce(
        [
            xx / max(1, width - 1),
            1 - xx / max(1, width - 1),
            yy / max(1, height - 1),
            1 - yy / max(1, height - 1),
        ]
    )
    return np.clip(edge * strength, 0, 1)


def suppress_gridlike_mask(mask):
    cleaned = np.clip(mask.astype(np.float32), 0, 1).copy()
    if cleaned.size == 0:
        return cleaned
    h, w = cleaned.shape
    if w >= 8:
        col_profile = cleaned.mean(axis=0)
        col_cut = max(0.32, float(np.percentile(col_profile, 90)))
        grid_cols = col_profile > col_cut
        if np.any(grid_cols):
            kernel = np.ones(3, dtype=np.uint8)
            grid_cols = np.convolve(grid_cols.astype(np.uint8), kernel, mode="same") > 0
            cleaned[:, grid_cols] *= 0.26
    if h >= 8:
        row_profile = cleaned.mean(axis=1)
        row_cut = max(0.32, float(np.percentile(row_profile, 90)))
        grid_rows = row_profile > row_cut
        if np.any(grid_rows):
            kernel = np.ones(3, dtype=np.uint8)
            grid_rows = np.convolve(grid_rows.astype(np.uint8), kernel, mode="same") > 0
            cleaned[grid_rows, :] *= 0.26
    return np.clip(cleaned, 0, 1)


def crack_patch_texture_mask(patch, size, seed):
    width, height = size
    if patch is None or width <= 1 or height <= 1:
        return None, None

    rgb = patch[:, :, :3] if patch.ndim == 3 and patch.shape[2] >= 3 else patch
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    low, high = np.percentile(gray, [3, 98])
    gray_norm = np.clip((gray - low) / max(1.0, high - low), 0, 1)

    if patch.ndim == 3 and patch.shape[2] == 4:
        alpha = cv2.resize(patch[:, :, 3].astype(np.float32) / 255.0, (width, height), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.GaussianBlur(np.clip(alpha, 0, 1), (0, 0), max(0.25, min(width, height) * 0.012))
    else:
        alpha = None

    blur = cv2.GaussianBlur(gray_norm, (0, 0), max(0.9, min(width, height) * 0.085))
    contrast = np.abs(gray_norm - blur)
    edge = np.abs(cv2.Laplacian(gray_norm.astype(np.float32), cv2.CV_32F))
    bright = np.clip((gray_norm - np.percentile(gray_norm, 72)) / 0.24, 0, 1)
    dark = np.clip((np.percentile(gray_norm, 30) - gray_norm) / 0.22, 0, 1)
    candidate = np.maximum.reduce(
        [
            bright * 0.95,
            dark * 0.48,
            np.clip(contrast * 3.0, 0, 1),
            np.clip(edge * 2.3, 0, 1),
        ]
    )
    if alpha is not None and np.max(alpha) > 0.08:
        candidate = np.maximum(candidate * 0.48, alpha)

    threshold = np.percentile(candidate, 78)
    mask = np.clip((candidate - threshold) / max(0.08, 1.0 - threshold), 0, 1)
    texture = smooth_unit_field((height, width), seed, max(1.0, min(width, height) * 0.14), octaves=2)
    mask = np.clip(mask * (0.78 + texture * 0.36), 0, 1)
    mask = suppress_gridlike_mask(mask)
    mask *= cell_edge_feather(width, height, strength=9.5)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.25, min(width, height) * 0.010))
    return gray_norm, np.clip(mask, 0, 1)


def apply_real_thermal_fault_patch(temp, cell, fault_type, scale):
    fault_type = canonical_fault_type(fault_type)
    if fault_type in SURFACE_OBSTRUCTION_FAULTS:
        return False
    rng = np.random.default_rng(seed_for(cell["id"], fault_type, scale, "real-patch"))
    patch = selected_fault_patch(fault_type, seed_for(cell["id"], fault_type, scale, "patch-choice"))
    if patch is None:
        return False

    x1, y1, x2, y2 = fault_render_bounds_for_cell(cell, fault_type)
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = 0 if w < 9 else max(1, int(w * 0.06))
    pad_y = 0 if h < 9 else max(1, int(h * 0.08))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    if bx2 <= bx1 or by2 <= by1:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    severity = clamp(scale, 1, 10) / 10
    boxes = real_fault_patch_boxes(fault_type, bx1, by1, bx2, by2, rng)
    for idx, box in enumerate(boxes):
        px1, py1, px2, py2 = box
        pw, ph = max(2, px2 - px1), max(2, py2 - py1)
        if fault_type in CRACKING_FAULTS:
            field, alpha = crack_patch_texture_mask(patch, (pw, ph), seed_for(cell["id"], fault_type, idx, "crack-mask"))
        else:
            field, alpha = normalized_patch_field(patch, (pw, ph), seed_for(cell["id"], fault_type, idx, "patch-field"))
        if field is None:
            continue
        organic = organic_fault_mask(pw, ph, fault_type, seed_for(cell["id"], fault_type, idx, "organic-mask"))
        if fault_type in HOTSPOT_FAULTS:
            field = np.clip(field * (0.36 + organic * 0.92), 0, 1)
            alpha = np.clip(np.clip((field - 0.42) / 0.46, 0, 1) * (0.45 + organic * 0.88), 0, 1)
        elif fault_type in CRACKING_FAULTS:
            alpha = np.clip(alpha * (0.76 + organic * 0.42), 0, 1)
        else:
            field = np.clip(field * (0.58 + organic * 0.72), 0, 1)
            alpha = np.clip(alpha * (0.32 + organic * 1.05), 0, 1)

        region = temp[py1:py2, px1:px2]
        if region.shape[:2] != field.shape:
            continue

        if fault_type in HOTSPOT_FAULTS:
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            cx = pw * rng.uniform(0.42, 0.58)
            cy = ph * rng.uniform(0.42, 0.58)
            rx = max(1.0, pw * rng.uniform(0.16, 0.26))
            ry = max(1.0, ph * rng.uniform(0.18, 0.30))
            core = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 2.2)
            hot_islands = np.clip(np.maximum(core, field * organic) - 0.18, 0, 1)
            patch_alpha = np.clip(alpha * 0.42 + hot_islands * 0.68, 0, 0.88)
            lift = hot_islands * (46 + 70 * severity) + core * (74 + 92 * severity)
            target = np.maximum(region + lift, 238 + hot_islands * (38 + 44 * severity))
            cool_base = region - (1 - patch_alpha) * (4 + 8 * severity)
            temp[py1:py2, px1:px2] = cool_base * (1 - patch_alpha) + target * patch_alpha
        elif fault_type in SURFACE_OBSTRUCTION_FAULTS:
            patch_alpha = np.clip(np.maximum(alpha, organic * 0.82) * (0.78 + 0.30 * severity), 0, 0.94)
            cool = patch_alpha * (48 + 54 * severity)
            rim = cv2.Laplacian(patch_alpha.astype(np.float32), cv2.CV_32F)
            warm_edge = np.clip(rim, 0, 1) * (82 + 86 * severity)
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            hot = np.zeros((ph, pw), dtype=np.float32)
            for _ in range(2):
                cx = pw * rng.uniform(0.20, 0.58)
                cy = ph * rng.uniform(0.28, 0.72)
                rx = max(1.0, pw * rng.uniform(0.18, 0.30))
                ry = max(1.0, ph * rng.uniform(0.22, 0.38))
                hot = np.maximum(hot, np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 1.35))
            temp[py1:py2, px1:px2] = np.maximum(region - cool + warm_edge, region + hot * (92 + 118 * severity))
        elif fault_type in CRACKING_FAULTS:
            cx = pw * rng.uniform(0.44, 0.56)
            cy = ph * rng.uniform(0.42, 0.58)
            angle = rng.uniform(-0.70, 0.70)
            if ph > pw:
                angle += math.pi / 2

            path = []
            bend = 0.0
            bend_velocity = rng.normal(0, 0.030)
            length = rng.uniform(0.58, 0.76)
            wave = rng.uniform(1.45, 2.75)
            phase = rng.uniform(-math.pi, math.pi)
            for point_idx in range(11):
                t = point_idx / 10.0 - 0.5
                bend_velocity += rng.normal(0, 0.035)
                bend = np.clip(bend + bend_velocity, -0.150, 0.150)
                bend += 0.050 * math.sin((t + 0.5) * math.pi * wave + phase)
                along = t * length
                px = cx + along * pw * math.cos(angle) - bend * pw * math.sin(angle)
                py = cy + along * ph * math.sin(angle) + bend * ph * math.cos(angle)
                path.append((clamp(int(round(px)), 1, pw - 2), clamp(int(round(py)), 1, ph - 2)))

            core_mask = np.zeros((ph, pw), dtype=np.float32)
            hot_mask = np.zeros((ph, pw), dtype=np.float32)
            warm_mask = np.zeros((ph, pw), dtype=np.float32)
            points = np.array(path, dtype=np.int32)
            core_width_px = max(1, int(round(min(pw, ph) * rng.uniform(0.020, 0.030))))
            hot_width_px = max(core_width_px + 2, int(round(min(pw, ph) * rng.uniform(0.090, 0.130))))
            warm_width_px = max(hot_width_px + 3, int(round(min(pw, ph) * rng.uniform(0.220, 0.300))))
            cv2.polylines(core_mask, [points], False, 1.0, core_width_px, lineType=cv2.LINE_AA)
            cv2.polylines(hot_mask, [points], False, 1.0, hot_width_px, lineType=cv2.LINE_AA)
            cv2.polylines(warm_mask, [points], False, 1.0, warm_width_px, lineType=cv2.LINE_AA)

            core = cv2.GaussianBlur(core_mask, (0, 0), max(0.35, core_width_px * 0.45))
            hot = cv2.GaussianBlur(hot_mask, (0, 0), max(0.65, hot_width_px * 0.40))
            warm = cv2.GaussianBlur(warm_mask, (0, 0), max(0.90, warm_width_px * 0.45))
            hot_ring = np.clip(hot - core * 0.48, 0, 1)
            warm_halo = np.clip(warm - hot * 0.40, 0, 1)

            texture = np.clip(0.74 + 0.20 * field + 0.10 * organic, 0.65, 1.0)
            core = np.clip(core * (0.90 + 0.10 * texture), 0, 1)
            hot_ring = np.clip(hot_ring * texture, 0, 1)
            warm_halo = np.clip(warm_halo * (0.92 + 0.08 * texture), 0, 1)

            cool_core = 86 + field * 18 - core * (10 + 12 * severity)
            warm_target = region + warm_halo * (64 + 58 * severity) + hot_ring * (124 + 130 * severity)
            warm_target = np.maximum(warm_target, 226 + warm_halo * (42 + 42 * severity) + hot_ring * (64 + 64 * severity))

            warm_alpha = np.clip(warm_halo * 0.70 + hot_ring * 0.94, 0, 0.97)
            cracked = region * (1 - warm_alpha) + warm_target * warm_alpha
            core_alpha = np.clip(core * (0.88 + 0.10 * severity), 0, 0.96)
            temp[py1:py2, px1:px2] = cracked * (1 - core_alpha) + cool_core * core_alpha
        elif fault_type in SOILING_FAULTS:
            soil = np.clip(np.maximum(alpha * 0.92, organic * 0.70), 0, 1)
            patch_alpha = np.clip(soil * (0.84 + 0.20 * severity), 0, 0.96)
            cool_target = np.minimum(region - soil * (62 + 70 * severity), 82 + field * (16 + 12 * severity))
            rim = cv2.GaussianBlur(np.abs(cv2.Laplacian(soil.astype(np.float32), cv2.CV_32F)), (0, 0), 0.55)
            warm_edge = np.clip(rim, 0, 1) * (24 + 32 * severity)
            temp[py1:py2, px1:px2] = region * (1 - patch_alpha) + cool_target * patch_alpha + warm_edge
        elif fault_type in DIODE_FAULTS:
            band = np.clip((field - 0.10) / 0.58, 0, 1)
            patch_alpha = np.clip(np.maximum(alpha * 0.52, band * (0.78 + 0.24 * severity)), 0, 0.94)
            lift = (0.40 + band * 0.95) * (92 + 104 * severity)
            target = np.maximum(region + lift, 240 + band * (50 + 52 * severity))
            temp[py1:py2, px1:px2] = region * (1 - patch_alpha) + target * patch_alpha
        elif fault_type in BYPASSED_FAULTS | THERMAL_BLOCK_FAULTS:
            patch_alpha = np.clip(np.clip((field - 0.15) / 0.58, 0, 1) * (0.62 + 0.28 * severity), 0, 0.86)
            lift = (0.30 + field * 0.88) * (74 + 82 * severity)
            target = np.maximum(region + lift, 232 + field * (40 + 42 * severity))
            temp[py1:py2, px1:px2] = region * (1 - patch_alpha) + target * patch_alpha
        elif fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
            patch_alpha = np.clip(np.clip((field - 0.16) / 0.62, 0, 1) * (0.58 + 0.24 * severity), 0, 0.78)
            lift = (0.26 + field * 0.82) * (66 + 76 * severity)
            cool = (1 - patch_alpha) * (5 + 8 * severity) if fault_type == "StringOpenCircuit" else 0
            target = np.maximum(region + lift, 220 + field * (20 + 30 * severity))
            temp[py1:py2, px1:px2] = (region - cool) * (1 - patch_alpha) + target * patch_alpha
        else:
            patch_alpha = np.clip(alpha * (0.45 + 0.25 * severity), 0, 0.78)
            lift = (0.25 + field * 0.75) * (24 + 44 * severity)
            temp[py1:py2, px1:px2] = region * (1 - patch_alpha) + (region + lift) * patch_alpha
    return True


def fallback_fault_tint_field(width, height, fault_type, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if fault_type in HOTSPOT_FAULTS:
        cx = width * rng.uniform(0.38, 0.62)
        cy = height * rng.uniform(0.35, 0.65)
        rx = max(1.0, width * rng.uniform(0.20, 0.34))
        ry = max(1.0, height * rng.uniform(0.24, 0.40))
        field = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 1.15)
    elif fault_type in DIODE_FAULTS:
        center = height * rng.uniform(0.42, 0.58)
        band = max(1.0, height * rng.uniform(0.18, 0.28))
        field = np.exp(-((yy - center) / band) ** 2)
    elif fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
        center = width * rng.uniform(0.42, 0.58)
        band = max(1.0, width * rng.uniform(0.16, 0.24))
        field = np.exp(-((xx - center) / band) ** 2)
    else:
        field = smooth_unit_field((height, width), seed, max(1.0, min(width, height) * 0.20), octaves=2)
    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), max(0.45, min(width, height) * 0.055))
    return np.clip(field, 0, 1)


def thermal_fault_tint_mask(shape, cell, fault_type, scale):
    fault_type = canonical_fault_type(fault_type)
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = fault_render_bounds_for_cell(cell, fault_type)
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = 0 if w < 9 else max(1, int(w * 0.06))
    pad_y = 0 if h < 9 else max(1, int(h * 0.08))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    if bx2 <= bx1 or by2 <= by1:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    rng = np.random.default_rng(seed_for(cell["id"], fault_type, scale, "thermal-tint"))
    patch = selected_fault_patch(fault_type, seed_for(cell["id"], fault_type, scale, "tint-patch-choice"))
    boxes = real_fault_patch_boxes(fault_type, bx1, by1, bx2, by2, rng)
    for idx, box in enumerate(boxes):
        px1, py1, px2, py2 = box
        px1, px2 = clamp(px1, bx1, bx2), clamp(px2, bx1, bx2)
        py1, py2 = clamp(py1, by1, by2), clamp(py2, by1, by2)
        if px2 <= px1 or py2 <= py1:
            continue
        local_w, local_h = px2 - px1, py2 - py1
        field = None
        if patch is not None:
            field, _ = normalized_patch_field(patch, (local_w, local_h), seed_for(cell["id"], fault_type, idx, "tint-field"))
        if field is None:
            field = fallback_fault_tint_field(local_w, local_h, fault_type, seed_for(cell["id"], fault_type, idx, "fallback-tint"))
        organic = organic_fault_mask(local_w, local_h, fault_type, seed_for(cell["id"], fault_type, idx, "tint-organic-mask"))
        if fault_type in HOTSPOT_FAULTS:
            field = np.clip(field * (0.34 + organic * 0.96), 0, 1)
            alpha = np.clip((field - 0.38) / 0.42, 0, 1) * np.clip(0.22 + organic * 0.82, 0, 1)
        else:
            field = np.clip(field * (0.52 + organic * 0.78), 0, 1)
            alpha = np.clip((field - 0.08) / 0.66, 0, 1) * np.clip(0.40 + organic * 0.96, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.35, min(local_w, local_h) * 0.04))
        mask[py1:py2, px1:px2] = np.maximum(mask[py1:py2, px1:px2], alpha[:local_h, :local_w])

    # Absolute clip to the selected cell/module interior.
    cell_mask = np.zeros_like(mask)
    cv2.rectangle(cell_mask, (x1, y1), (x2 - 1, y2 - 1), 1.0, -1)
    preserve = panel_segmentation_preserve_mask(shape, cell)
    return np.clip(mask * cell_mask * (1.0 - preserve * 0.68), 0, 1)


def surface_obstruction_shape(width, height, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.float32)
    if width <= 2 or height <= 2:
        return mask

    top_left = (int(width * rng.uniform(-0.08, 0.18)), int(height * rng.uniform(0.04, 0.22)))
    top_right = (int(width * rng.uniform(0.54, 1.08)), int(height * rng.uniform(0.16, 0.38)))
    bottom_right = (int(width * rng.uniform(0.38, 0.98)), int(height * rng.uniform(0.58, 0.96)))
    bottom_left = (int(width * rng.uniform(-0.10, 0.22)), int(height * rng.uniform(0.42, 0.84)))
    cv2.fillPoly(mask, [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)], 0.78)

    leaf_count = int(rng.integers(2, 5))
    for _ in range(leaf_count):
        cx = int(width * rng.uniform(0.12, 0.78))
        cy = int(height * rng.uniform(0.18, 0.82))
        rx = max(1, int(width * rng.uniform(0.08, 0.18)))
        ry = max(1, int(height * rng.uniform(0.12, 0.28)))
        cv2.ellipse(mask, (cx, cy), (rx, ry), float(rng.uniform(-55, 55)), 0, 360, 0.92, -1, cv2.LINE_AA)

    texture = smooth_unit_field(
        (height, width),
        seed_for(seed, "surface-obstruction-texture"),
        max(1.0, min(width, height) * 0.16),
        octaves=3,
    )
    mask *= np.clip(0.62 + texture * 0.42, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(0.45, min(width, height) * 0.030))
    mask = np.where(mask > 0.065, mask, 0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(0.30, min(width, height) * 0.010))
    return np.clip(mask, 0, 0.95)


def apply_surface_obstruction_rgb_tint(output, cell, scale):
    height, width = output.shape[:2]
    x1, y1, x2, y2 = fault_render_bounds_for_cell(cell, "PartialShading")
    x1, x2 = clamp(x1, 0, width), clamp(x2, 0, width)
    y1, y2 = clamp(y1, 0, height), clamp(y2, 0, height)
    if x2 <= x1 or y2 <= y1:
        return output

    local_w, local_h = x2 - x1, y2 - y1
    severity = clamp(scale, 1, 10) / 10
    shadow = surface_obstruction_shape(local_w, local_h, seed_for(cell["id"], scale, "surface-obstruction-rgb"))

    edge = np.clip(cv2.Laplacian(shadow.astype(np.float32), cv2.CV_32F), 0, 1)
    hot = cv2.GaussianBlur(edge, (0, 0), max(0.35, min(local_w, local_h) * 0.012))
    hot = np.clip(hot * 0.24, 0, 0.22)

    preserve = panel_segmentation_preserve_mask(output.shape, cell)[y1:y2, x1:x2]
    visible_shadow = np.clip(shadow * (1.0 - preserve * 0.72), 0, 1)
    visible_hot = np.clip(hot * (1.0 - preserve * 0.50), 0, 1)

    region = output[y1:y2, x1:x2]
    cool_color = np.array([62, 18, 122], dtype=np.float32)
    warm_color = np.array([214, 88, 38], dtype=np.float32)
    white_hot = np.array([230, 128, 62], dtype=np.float32)

    cool_alpha = (visible_shadow * (0.42 + 0.16 * severity))[..., None]
    region = region * (1 - cool_alpha) + cool_color * cool_alpha
    warm_alpha = (visible_hot * (0.12 + 0.05 * severity))[..., None]
    region = region * (1 - warm_alpha) + warm_color * warm_alpha
    core_alpha = (np.clip((visible_hot - 0.34) / 0.30, 0, 1) * (0.02 + 0.03 * severity))[..., None]
    region = region * (1 - core_alpha) + white_hot * core_alpha
    output[y1:y2, x1:x2] = region
    return output


def apply_cell_cracking_rgb_tint(output, cell, scale):
    height, width = output.shape[:2]
    x1, y1, x2, y2 = fault_render_bounds_for_cell(cell, "CellCracking")
    x1, x2 = clamp(x1, 0, width), clamp(x2, 0, width)
    y1, y2 = clamp(y1, 0, height), clamp(y2, 0, height)
    if x2 <= x1 or y2 <= y1:
        return output

    local_w, local_h = x2 - x1, y2 - y1
    severity = clamp(scale, 1, 10) / 10

    patch = selected_fault_patch("CellCracking", seed_for(cell["id"], scale, "cell-crack-patch"))
    field = None
    real_mask = None
    if patch is not None:
        field, real_mask = crack_patch_texture_mask(patch, (local_w, local_h), seed_for(cell["id"], scale, "cell-crack-field"))
    if field is None:
        field = smooth_unit_field((local_h, local_w), seed_for(cell["id"], scale, "cell-crack-fallback"), max(1.0, min(local_w, local_h) * 0.14), octaves=3)
        real_mask = np.clip((field - np.percentile(field, 78)) / 0.24, 0, 1)
    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), max(0.45, min(local_w, local_h) * 0.035))
    if real_mask is not None:
        real_mask = cv2.GaussianBlur(np.clip(real_mask.astype(np.float32), 0, 1), (0, 0), max(0.35, min(local_w, local_h) * 0.018))

    bright_fracture = np.clip((field - 0.68) / 0.24, 0, 1)
    crack = np.clip(np.maximum(real_mask, bright_fracture * 0.42), 0, 1)
    edge_source = crack
    edge = np.abs(cv2.Laplacian(edge_source.astype(np.float32), cv2.CV_32F))
    edge = cv2.GaussianBlur(np.clip(edge * 2.2, 0, 1), (0, 0), max(0.35, min(local_w, local_h) * 0.018))
    bloom = cv2.GaussianBlur(crack.astype(np.float32), (0, 0), max(0.9, min(local_w, local_h) * 0.110))
    crack = cv2.GaussianBlur(np.clip(crack, 0, 1), (0, 0), max(0.35, min(local_w, local_h) * 0.014))
    hot = np.clip(edge * 0.32 + bloom * (0.82 + 0.18 * severity), 0, 1)

    preserve = panel_segmentation_preserve_mask(output.shape, cell)[y1:y2, x1:x2]
    visible_crack = np.clip(crack * (1.0 - preserve * 0.50), 0, 1)
    visible_hot = np.clip(hot * (1.0 - preserve * 0.38), 0, 1)

    region = output[y1:y2, x1:x2]
    cool_color = np.array([72, 12, 112], dtype=np.float32)
    warm_color = np.array([242, 126, 44], dtype=np.float32)
    white_hot = np.array([255, 218, 118], dtype=np.float32)
    cool_alpha = (np.clip(visible_crack - visible_hot * 0.20, 0, 1) * (0.24 + 0.12 * severity))[..., None]
    region = region * (1 - cool_alpha) + cool_color * cool_alpha
    warm_alpha = (visible_hot * (0.70 + 0.18 * severity))[..., None]
    region = region * (1 - warm_alpha) + warm_color * warm_alpha
    core_alpha = (np.clip((visible_hot - 0.52) / 0.30, 0, 1) * (0.26 + 0.14 * severity))[..., None]
    region = region * (1 - core_alpha) + white_hot * core_alpha
    output[y1:y2, x1:x2] = region
    return output


def apply_thermal_fault_tints(thermal_rgb, tint_items):
    if not tint_items:
        return thermal_rgb
    output = thermal_rgb.astype(np.float32)
    for cell, fault_type, scale in tint_items:
        if canonical_fault_type(fault_type) in SURFACE_OBSTRUCTION_FAULTS:
            output = apply_surface_obstruction_rgb_tint(output, cell, scale)
            continue
        if canonical_fault_type(fault_type) in CRACKING_FAULTS:
            output = apply_cell_cracking_rgb_tint(output, cell, scale)
            continue
        mask = thermal_fault_tint_mask(output.shape, cell, fault_type, scale)
        if not np.any(mask > 0.01):
            continue
        severity = clamp(scale, 1, 10) / 10
        if canonical_fault_type(fault_type) in HOTSPOT_FAULTS:
            color = np.array([255, 218, 72], dtype=np.float32)
            strength = 0.88 + 0.12 * severity
        elif canonical_fault_type(fault_type) in DIODE_FAULTS:
            color = np.array([255, 205, 62], dtype=np.float32)
            strength = 0.90 + 0.14 * severity
        elif canonical_fault_type(fault_type) in BYPASSED_FAULTS:
            color = np.array([252, 166, 48], dtype=np.float32)
            strength = 0.78 + 0.16 * severity
        elif canonical_fault_type(fault_type) in STRING_FAULTS | LEGACY_LINE_FAULTS:
            color = np.array([245, 138, 44], dtype=np.float32)
            strength = 0.62 + 0.18 * severity
        elif canonical_fault_type(fault_type) in SOILING_FAULTS:
            color = np.array([52, 24, 96], dtype=np.float32)
            strength = 0.84 + 0.18 * severity
        else:
            color = np.array([248, 154, 46], dtype=np.float32)
            strength = 0.64 + 0.16 * severity
        core = np.clip((mask - 0.42) / 0.38, 0, 1)
        rim = np.clip(mask - core * 0.72, 0, 1)
        cool = np.array([74, 16, 102], dtype=np.float32)
        rim_alpha = (rim * 0.18)[..., None]
        output = output * (1 - rim_alpha) + cool * rim_alpha
        alpha = (mask * strength)[..., None]
        output = output * (1 - alpha) + color * alpha
        if canonical_fault_type(fault_type) in HOTSPOT_FAULTS:
            white_alpha = (core * (0.34 + 0.22 * severity))[..., None]
        elif canonical_fault_type(fault_type) in DIODE_FAULTS:
            white_alpha = (core * (0.28 + 0.18 * severity))[..., None]
        elif canonical_fault_type(fault_type) in SOILING_FAULTS:
            white_alpha = (core * (0.03 + 0.03 * severity))[..., None]
        else:
            white_alpha = (core * (0.10 + 0.08 * severity))[..., None]
        output = output * (1 - white_alpha) + np.array([255, 246, 158], dtype=np.float32) * white_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_thermal_fault(temp, cell, fault_type, scale):
    if cell is None:
        return temp
    fault_type = canonical_fault_type(fault_type)
    if apply_real_thermal_fault_patch(temp, cell, fault_type, scale):
        return temp
    x1, y1, x2, y2 = fault_render_bounds_for_cell(cell, fault_type)
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = 0 if w < 9 else max(1, int(w * 0.06))
    pad_y = 0 if h < 9 else max(1, int(h * 0.08))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    if bx2 <= bx1 or by2 <= by1:
        bx1, by1, bx2, by2 = x1, y1, x2, y2
    bw, bh = max(3, bx2 - bx1), max(3, by2 - by1)
    severity = clamp(scale, 1, 10) / 10
    rng = np.random.default_rng(seed_for(cell["id"], fault_type, scale, "thermal-field"))

    if fault_type in HOTSPOT_FAULTS:
        spot_count = 1 if fault_type == "SingleHotSpot" else int(rng.integers(2, 5))
        for spot_idx in range(spot_count):
            center_x = bx1 + bw * rng.uniform(0.28, 0.72)
            center_y = by1 + bh * rng.uniform(0.30, 0.70)
            radius_x = max(1.4, bw * rng.uniform(0.22, 0.34) * (0.75 + 0.50 * severity))
            radius_y = max(1.2, bh * rng.uniform(0.24, 0.38) * (0.75 + 0.50 * severity))
            spread = max(radius_x, radius_y) * 3.8
            hx1 = clamp(int(center_x - spread), bx1, bx2 - 1)
            hy1 = clamp(int(center_y - spread), by1, by2 - 1)
            hx2 = clamp(int(center_x + spread) + 1, hx1 + 1, bx2)
            hy2 = clamp(int(center_y + spread) + 1, hy1 + 1, by2)
            local_w, local_h = hx2 - hx1, hy2 - hy1
            yy, xx = np.mgrid[0:local_h, 0:local_w].astype(np.float32)
            dx = xx + hx1 - center_x
            dy = yy + hy1 - center_y
            angle = rng.uniform(-0.9, 0.9)
            xr = np.cos(angle) * dx + np.sin(angle) * dy
            yr = -np.sin(angle) * dx + np.cos(angle) * dy
            dist = (xr / radius_x) ** 2 + (yr / radius_y) ** 2
            halo = np.exp(-dist * 0.55)
            core = np.exp(-dist * 2.9)
            rough = smooth_unit_field((local_h, local_w), seed_for(cell["id"], scale, spot_idx, "hotspot-edge"), max(1.0, min(local_w, local_h) * 0.20), octaves=3)
            mask = np.clip(halo * (0.78 + rough * 0.35), 0, 1)
            region = temp[hy1:hy2, hx1:hx2]
            lift = mask * (12 + 24 * severity) + core * (54 + 74 * severity)
            target = np.maximum(region + lift, 246 + core * (18 + 36 * severity))
            spot_alpha = np.clip((mask - 0.035) / 0.58, 0, 1)
            temp[hy1:hy2, hx1:hx2] = region * (1.0 - spot_alpha) + target * spot_alpha

    elif fault_type in DIODE_FAULTS:
        band_count = 1 if fault_type in {"SingleDiode", "DB"} else 2
        for band_idx in range(band_count):
            local_by1 = by1 + int((band_idx - (band_count - 1) / 2) * bh * 0.34)
            local_by2 = by2 + int((band_idx - (band_count - 1) / 2) * bh * 0.34)
            local_by1 = clamp(local_by1, y1, y2 - 2)
            local_by2 = clamp(local_by2, local_by1 + 2, y2)
            local_bh = max(3, local_by2 - local_by1)
            band_h = max(2, int(local_bh * (0.18 + 0.15 * severity)))
            cy = local_by1 + local_bh // 2
            band = np.zeros((local_bh, bw), dtype=np.float32)
            mid = cy - local_by1
            pts = np.array(
                [
                    [0, clamp(mid - band_h // 2 + int(rng.integers(-2, 3)), 0, local_bh - 1)],
                    [bw - 1, clamp(mid - band_h // 2 + int(rng.integers(-2, 3)), 0, local_bh - 1)],
                    [bw - 1, clamp(mid + band_h // 2 + int(rng.integers(-2, 3)), 0, local_bh - 1)],
                    [0, clamp(mid + band_h // 2 + int(rng.integers(-2, 3)), 0, local_bh - 1)],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(band, [pts], 1.0)
            band *= pv_environment_multiplier((local_bh, bw), seed_for(cell["id"], band_idx, "diode-band-texture"), strength=0.22)
            band = cv2.GaussianBlur(band, (0, 0), max(0.7, band_h * 0.22))
            temp[local_by1:local_by2, bx1:bx2] += band[:local_bh, :bw] * (36 + 52 * severity)

    elif fault_type in BYPASSED_FAULTS:
        section_count = 1 if fault_type == "SingleByPassed" else 2
        section_w = max(3, int(bw * rng.uniform(0.24, 0.38)))
        for section_idx in range(section_count):
            sx = bx1 + int((section_idx + 0.5) * bw / section_count - section_w / 2 + rng.normal(0, bw * 0.06))
            sx = clamp(sx, bx1, bx2 - section_w)
            local = temp[by1:by2, sx:sx + section_w]
            field = smooth_unit_field(local.shape, seed_for(cell["id"], section_idx, "bypassed-field"), max(1.0, min(local.shape) * 0.22), octaves=2)
            contrast = (field - 0.45) * (24 + 28 * severity)
            local[:, :] = np.clip(local + contrast + (10 + 24 * severity), 0, 255)

    elif fault_type in STRING_FAULTS:
        fault = np.zeros((bh, bw), dtype=np.float32)
        if fault_type == "StringOpenCircuit":
            stripe_x = int(bw * rng.uniform(0.35, 0.65))
            cv2.line(fault, (stripe_x, 0), (stripe_x + int(rng.normal(0, 2)), bh - 1), 1.0, max(1, int(bw * 0.18)))
            temp[by1:by2, bx1:bx2] -= (1 - fault[:bh, :bw]) * (6 + 12 * severity)
        else:
            cv2.line(fault, (0, int(bh * 0.30)), (bw - 1, int(bh * 0.68)), 1.0, max(1, int(min(bw, bh) * 0.18)))
            cv2.line(fault, (0, int(bh * 0.70)), (bw - 1, int(bh * 0.32)), 0.65, max(1, int(min(bw, bh) * 0.10)))
        fault = cv2.GaussianBlur(fault, (0, 0), max(0.65, min(bw, bh) * 0.12))
        temp[by1:by2, bx1:bx2] += fault[:bh, :bw] * (42 + 54 * severity)

    elif fault_type in LEGACY_LINE_FAULTS:
        fault = np.zeros((bh, bw), dtype=np.float32)
        points = []
        for idx in range(6):
            t = idx / 5
            points.append((int(bw * t), clamp(int(bh * (0.18 + 0.64 * rng.random())), 0, bh - 1)))
        cv2.polylines(fault, [np.array(points, dtype=np.int32)], False, 1.0, max(1, int(min(bw, bh) * 0.12)))
        fault = cv2.GaussianBlur(fault, (0, 0), max(0.5, min(bw, bh) * 0.06))
        temp[by1:by2, bx1:bx2] += fault[:bh, :bw] * (34 + 46 * severity)

    elif fault_type in THERMAL_BLOCK_FAULTS:
        mask = np.array(irregular_mask((bw, bh), seed_for(cell["id"], "thermal-block-mask"), blobs=5), dtype=np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.8, min(bw, bh) * 0.08))
        temp[by1:by2, bx1:bx2] += mask[:bh, :bw] * (32 + 54 * severity)

    elif fault_type == "Bypass diode":
        band_h = max(2, int(bh * (0.22 + 0.20 * severity)))
        cy = by1 + bh // 2
        band = np.zeros((bh, bw), dtype=np.float32)
        mid = cy - by1
        pts = np.array(
            [
                [0, clamp(mid - band_h // 2 + int(rng.integers(-2, 3)), 0, bh - 1)],
                [bw - 1, clamp(mid - band_h // 2 + int(rng.integers(-2, 3)), 0, bh - 1)],
                [bw - 1, clamp(mid + band_h // 2 + int(rng.integers(-2, 3)), 0, bh - 1)],
                [0, clamp(mid + band_h // 2 + int(rng.integers(-2, 3)), 0, bh - 1)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(band, [pts], 1.0)
        band *= pv_environment_multiplier((bh, bw), seed_for(cell["id"], "bypass-band-texture"), strength=0.18)
        band = cv2.GaussianBlur(band, (0, 0), max(0.7, band_h * 0.18))
        temp[by1:by2, bx1:bx2] += band[:bh, :bw] * (44 + 54 * severity)

    elif fault_type in SOILING_FAULTS | {"Bird droppings", "Snow cover"}:
        mask = np.array(irregular_mask((bw, bh), seed_for(cell["id"], "dust-mask"), blobs=9), dtype=np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.8, min(bw, bh) * 0.05))
        if fault_type == "Bird droppings":
            cool_delta = 38 + 30 * severity
            warm_rim = 18 + 18 * severity
        elif fault_type == "Snow cover":
            cool_delta = 54 + 42 * severity
            warm_rim = 6 + 8 * severity
        else:
            cool_delta = 34 + 44 * severity
            warm_rim = 14 + 18 * severity
        temp[by1:by2, bx1:bx2] -= mask[:bh, :bw] * cool_delta
        edge = cv2.Laplacian(mask, cv2.CV_32F)
        temp[by1:by2, bx1:bx2] += np.clip(edge, 0, 1)[:bh, :bw] * warm_rim

    elif fault_type in SURFACE_OBSTRUCTION_FAULTS:
        shade = surface_obstruction_shape(bw, bh, seed_for(cell["id"], scale, "surface-obstruction-temp"))
        local = temp[by1:by2, bx1:bx2]
        shade_local = shade[:bh, :bw]
        cool_target = 104 + 18 * (1 - shade_local)
        local[:] = local * (1 - shade_local * (0.50 + 0.10 * severity)) + cool_target * (shade_local * (0.50 + 0.10 * severity))
        edge = np.clip(cv2.Laplacian(shade, cv2.CV_32F), 0, 1)
        local += edge[:bh, :bw] * (8 + 14 * severity)

    elif fault_type == "Electrical / open circuit":
        fault = np.zeros((bh, bw), dtype=np.float32)
        cv2.line(fault, (bw // 2, 0), (bw // 2, bh - 1), 1.0, max(1, int(bw * 0.22)))
        cv2.line(fault, (0, bh // 2), (bw - 1, bh // 2), 0.55, max(1, int(bh * 0.16)))
        fault = cv2.GaussianBlur(fault, (0, 0), max(0.65, min(bw, bh) * 0.10))
        temp[by1:by2, bx1:bx2] += fault[:bh, :bw] * (48 + 52 * severity)
        temp[by1:by2, bx1:bx2] -= (1 - fault[:bh, :bw]) * (8 + 10 * severity)

    elif fault_type == "Degradation / resistance":
        gradient = np.linspace(0.15, 1.0, bw, dtype=np.float32)[None, :]
        waviness = rng.normal(0, 0.18, size=(bh, bw)).astype(np.float32)
        waviness = cv2.GaussianBlur(waviness, (0, 0), max(0.7, min(bw, bh) * 0.12))
        field = np.clip(gradient + waviness, 0, 1)
        temp[by1:by2, bx1:bx2] += field[:bh, :bw] * (18 + 28 * severity)

    else:
        crack = np.zeros((bh, bw), dtype=np.float32)
        points = []
        for idx in range(7):
            t = idx / 6
            px = int(bw * t)
            py = int(bh * (0.22 + 0.58 * t) + rng.integers(-max(1, bh // 5), max(2, bh // 5)))
            points.append((px, clamp(py, 0, bh - 1)))
        cv2.polylines(crack, [np.array(points, dtype=np.int32)], False, 1.0, max(1, int(min(bw, bh) * 0.10)))
        crack = cv2.GaussianBlur(crack, (0, 0), max(0.5, min(bw, bh) * 0.035))
        temp[by1:by2, bx1:bx2] -= crack[:bh, :bw] * (34 + 24 * severity)
        for px, py in points[1:-1:2]:
            r = max(1, int(min(bw, bh) * (0.10 + 0.04 * severity)))
            cv2.circle(temp[by1:by2, bx1:bx2], (px, py), r, 245 + 10 * severity, -1)

    return temp


def apply_sensor_model(temp, location_id):
    """Make the scalar temperature field behave more like a low-res thermal camera."""
    h, w = temp.shape
    rng = np.random.default_rng(seed_for(location_id, "sensor-model"))

    out = cv2.GaussianBlur(temp.astype(np.float32), (0, 0), 0.65)

    small = cv2.resize(out, (max(8, int(w * 0.72)), max(8, int(h * 0.72))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    small += rng.normal(0, 1.6, small.shape).astype(np.float32)
    small += rng.normal(0, 0.9, (1, sw)).astype(np.float32)
    small += rng.normal(0, 0.6, (sh, 1)).astype(np.float32)

    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    out = out - r2 * 4.0

    return np.clip(out, 0, 255)


def thermalize(image, location_id, include_pv, fault_cell=None, fault_type=None, fault_scale=DEFAULT_FAULT_SCALE, fault_records=None):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    frame = np.array(rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    soft = cv2.GaussianBlur(gray, (0, 0), 2.4)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(soft)

    terrain_low, terrain_high = {
        "desert_farm": (8, 82),
        "rooftop": (14, 92),
        "agri_rows": (12, 94),
        "farm_lake": (10, 82),
        "scenario_1": (12, 94),
    }.get(location_id, (12, 94))

    temp = cv2.normalize(clahe, None, terrain_low, terrain_high, cv2.NORM_MINMAX).astype(np.float32)

    broad = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 10)
    detail = gray.astype(np.float32) - broad
    temp += detail * 0.08

    rng = np.random.default_rng(seed_for(location_id, "opencv-thermal", include_pv))
    temp += thermal_noise(temp.shape, rng, 3.4, 0.70)
    temp += thermal_noise(temp.shape, rng, 7.2, 4.8)
    temp += thermal_noise(temp.shape, rng, 5.8, 14.0)

    scene_env = pv_environment_multiplier(
        temp.shape,
        seed_for(location_id, "thermal-cloud-sun", include_pv),
        strength=0.070,
    )
    temp *= scene_env
    temp = np.clip(temp, terrain_low - 8, terrain_high + 12)

    if include_pv:
        shadow_mask = np.zeros(temp.shape, dtype=np.uint8)

        for row in panels(location_id):
            row_h = max(1, row["y2"] - row["y1"])
            offset = max(2, int(row_h * 0.18))

            if custom_layout_enabled():
                module_count = custom_panels_per_row(location_id)
                for layout in module_layouts_for_row(row, module_count):
                    poly = module_visual_polygon(row, layout, "outer").copy()
                    poly[:, 0] += offset
                    poly[:, 1] += max(1, offset // 2)
                    poly[:, 0] = np.clip(poly[:, 0], 0, BASE_SIZE[0] - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, BASE_SIZE[1] - 1)
                    cv2.fillPoly(shadow_mask, [np.round(poly).astype(np.int32)], 180, cv2.LINE_AA)
            else:
                cv2.rectangle(
                    shadow_mask,
                    (row["x1"] + offset, row["y2"] - max(1, row_h // 12)),
                    (min(BASE_SIZE[0] - 1, row["x2"] + offset), min(BASE_SIZE[1] - 1, row["y2"] + offset)),
                    180,
                    -1,
                )

        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), 3.5).astype(np.float32) / 255.0
        shadow_strength = 5 if location_id == "desert_farm" else 14
        temp -= shadow_mask * shadow_strength

        for row in panels(location_id):
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            row_w = max(1, x2 - x1)
            row_h = max(1, y2 - y1)
            row_gray = gray[y1:y2, x1:x2]

            if row_gray.size == 0:
                continue

            panel_min, panel_max = {
                "desert_farm": (168, 200),
                "rooftop": (162, 194),
                "agri_rows": (164, 196),
                "farm_lake": (158, 190),
                "scenario_1": (164, 196),
            }.get(location_id, (164, 196))

            module_count = (
                custom_panels_per_row(location_id)
                if custom_layout_enabled()
                else max(
                    1,
                    len(cells_for_row(location_id, row["id"]))
                    // max(1, CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS),
                )
            )

            panel_local = procedural_thermal_panel_row(
                (row_w, row_h),
                module_count,
                seed_for(location_id, row["id"], "old-procedural-thermal-row"),
                panel_min,
                panel_max,
            )

            grad_x = np.linspace(-4, 4, row_w, dtype=np.float32)[None, :]
            grad_y = np.linspace(2, -2, row_h, dtype=np.float32)[:, None]
            panel_local += grad_x + grad_y
            panel_local = np.clip(panel_local, panel_min - 10, panel_max + 6).astype(np.float32)

            if custom_layout_enabled():
                for layout in module_layouts_for_row(row, module_count):
                    ox1, oy1, ox2, oy2 = layout["outer"]
                    lx1 = max(0, ox1 - x1)
                    ly1 = max(0, oy1 - y1)
                    lx2 = min(row_w, ox2 - x1)
                    ly2 = min(row_h, oy2 - y1)
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue
                    mrng = np.random.default_rng(
                        seed_for(location_id, row["id"], layout["module_number"], "module-temp")
                    )
                    panel_local[ly1:ly2, lx1:lx2] += float(mrng.normal(0, 3.0))

            row_mask = np.zeros((row_h, row_w), dtype=np.uint8)

            if custom_layout_enabled():
                for layout in module_layouts_for_row(row, module_count):
                    poly = module_visual_polygon(row, layout, "outer").copy()
                    poly[:, 0] -= x1
                    poly[:, 1] -= y1
                    cv2.fillPoly(row_mask, [np.round(poly).astype(np.int32)], 255, cv2.LINE_AA)
            else:
                cv2.rectangle(row_mask, (0, 0), (row_w - 1, row_h - 1), 255, -1)

            row_mask = cv2.GaussianBlur(row_mask, (0, 0), max(0.75, row_h * 0.045)).astype(np.float32) / 255.0
            row_mask = np.clip(row_mask * 0.98, 0, 0.98)

            temp[y1:y2, x1:x2] = temp[y1:y2, x1:x2] * (1.0 - row_mask) + panel_local * row_mask

            edge_temp = panel_min - 18

            if custom_layout_enabled():
                module_count = custom_panels_per_row(location_id)
                draw_thermal_module_cell_seams(temp, row, module_count, panel_min, include_string_grid=True)

                for layout in module_layouts_for_row(row, module_count):
                    outer = module_visual_polygon(row, layout, "outer")
                    left_top = tuple(np.round(outer[0]).astype(int))
                    left_bottom = tuple(np.round(outer[3]).astype(int))
                    if not actual_panel_model_enabled():
                        draw_wavy_line(
                            temp,
                            left_top,
                            left_bottom,
                            edge_temp - 30,
                            1,
                            seed_for(row["id"], layout["module_number"], "thermal-gap-left"),
                            0.32,
                        )
            else:
                draw_wavy_line(temp, (x1, y1), (x2, y1), edge_temp, 1, seed_for(row["id"], "thermal-row-top"), 0.35)
                draw_wavy_line(
                    temp,
                    (x1, y2),
                    (x2, y2),
                    edge_temp - 4,
                    max(1, row_h // 16),
                    seed_for(row["id"], "thermal-row-bottom"),
                    0.4,
                )
                draw_thermal_module_cell_seams(temp, row, module_count, panel_min, include_string_grid=True)

        if fault_records:
            for record in fault_records:
                row = row_by_id(location_id, record["row_id"])
                cell = cell_with_fault_bbox(
                    cell_by_id(location_id, row["id"], record["target_cell_id"]),
                    record,
                )
                temp = apply_thermal_fault(
                    temp,
                    cell,
                    record["fault_type"],
                    record.get("fault_scale", fault_scale),
                )
        elif fault_cell is not None and fault_type:
            temp = apply_thermal_fault(temp, fault_cell, fault_type, fault_scale)

        for row in panels(location_id):
            panel_min = {
                "desert_farm": 168,
                "rooftop": 162,
                "agri_rows": 164,
                "farm_lake": 158,
                "scenario_1": 164,
            }.get(location_id, 164)

            module_count = (
                custom_panels_per_row(location_id)
                if custom_layout_enabled()
                else max(
                    1,
                    len(cells_for_row(location_id, row["id"]))
                    // max(1, CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS),
                )
            )

            draw_thermal_module_cell_seams(temp, row, module_count, panel_min, include_string_grid=False)

    temp = apply_sensor_model(temp, location_id)

    dither = np.random.default_rng(seed_for(location_id, "thermal-dither")).uniform(
        -0.5,
        0.5,
        temp.shape,
    ).astype(np.float32)

    temp_u8 = np.clip(temp + dither, 0, 255).astype(np.uint8)
    lut = np.array(thermal_lut(), dtype=np.uint8)
    thermal_rgb = lut[temp_u8]

    return Image.fromarray(thermal_rgb)
