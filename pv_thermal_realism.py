"""Shared relative-temperature model, evaluated before faults and camera optics.

These are visual thermal scalars, not radiometric measurements. Spatial fields
represent irradiance, ventilation and material response. Geometry and fault
provenance are deliberately independent of this layer.
"""

THERMAL_REALISM_VERSION = "continuous-ir-reference-v13-materials-v12-frames"
THERMAL_SCALAR_DISPLAY_RANGE = (24.0, 254.0)


def thermal_coherent_field(shape, seed, grid=(7, 9)):
    """A Gaussian random field with a specified correlation length, not a grid.

    Filter on a bounded working plane, normalize once, then interpolate. No
    control-point cells, saturation plateaus, periodic subdivisions or blobs.
    """
    height, width = shape
    # A fixed latent plane makes a seed describe the same normalized field at
    # every camera resolution. Zooming cannot invent a different texture.
    wh, ww = 192, 192
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(0, 1, (wh, ww)).astype(np.float32)
    sigma_y = max(1.0, wh / max(2.0, float(grid[0])) * 0.48)
    sigma_x = max(1.0, ww / max(2.0, float(grid[1])) * 0.48)
    field = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y,
                             borderType=cv2.BORDER_REFLECT_101)
    field -= float(field.mean())
    field /= max(1e-6, float(field.std()))
    return cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)


def thermal_scene_drift(shape, seed):
    """Continuous scene exposure + smaller local ventilation/row drift."""
    height, width = shape
    broad = thermal_coherent_field(shape, seed_for(seed, "exposure"), (6, 8))
    local = thermal_coherent_field(shape, seed_for(seed, "ventilation"), (18, 24))
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    for start in range(0, height, 128):
        rows = slice(start, start + 128)
        broad[rows] = broad[rows] * 1.5 + local[rows] * 2.8 + x * 4.0 - y[rows] * 2.5
    return broad


def thermal_surface_microtexture(shape, seed):
    """Fixed scene-space surface roughness, with short correlated dark flecks.

    Separate from detector noise: this models material temperature differences.
    The fixed latent resolution keeps texture stable across scene resolutions.
    """
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(0, 1, (512, 512)).astype(np.float32)
    granular = cv2.GaussianBlur(raw, (0, 0), sigmaX=0.70, sigmaY=0.95)
    granular /= max(1e-6, float(granular.std()))
    cooling = np.maximum(-granular - 0.65, 0)
    # Continuous cool tails, not binary speckle masks or isolated stamped spots.
    roughness = granular * 1.7 - cooling * 2.2
    roughness -= float(roughness.mean())
    return cv2.resize(roughness, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def thermal_background_realism(base, image, location_id, material_masks=None):
    """Material/exposure response, independent of the module temperature range.

    RGB supplies material and shadow evidence, not output thermal colors. All
    temperatures use the same calibrated relative scalar range as the modules.
    """
    height, width = np.asarray(base).shape
    if max(height, width) > 1600:
        scale = 1600.0 / max(height, width)
        size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
        masks = None if material_masks is None else {
            key: cv2.resize(value, size, interpolation=cv2.INTER_NEAREST)
            for key, value in material_masks.items()
        }
        small = thermal_background_realism(
            np.empty((size[1], size[0]), np.float32), image.resize(size), location_id, masks
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    rgb = np.asarray((image if image.mode == "RGB" else image.convert("RGB")).resize((width, height)), dtype=np.uint8)
    red, green, blue = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = max(1.0, max(height, width) / 1600.0)
    luma = cv2.GaussianBlur(gray, (0, 0), 0.45 * scale) / 255.0
    seed = seed_for(location_id, "healthy-thermal-scene")
    drift = thermal_scene_drift((height, width), seed)
    broad = thermal_coherent_field((height, width), seed_for(seed, "soil-moisture"), (30, 40))
    fine = thermal_coherent_field((height, width), seed_for(seed, "surface-detail"), (96, 124))
    # Keep material structure and source detail, while reducing the finest
    # synthetic grain that otherwise competes with small PV modules in Far View.
    microtexture = thermal_surface_microtexture((height, width), seed_for(seed, "surface-roughness")) * 0.85
    vegetation = np.clip((2 * green - red - blue - 8) / 65.0, 0, 1)
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))
    neutral_surface = np.clip((28.0 - chroma) / 23.0, 0, 1)
    neutral_surface *= np.clip((luma - 0.14) / 0.24, 0, 1)
    water = np.clip((blue - red - 12) / 40.0, 0, 1) * np.clip((blue - green + 8) / 35.0, 0, 1)
    if location_id in {"floating_water", "farm_lake"}:
        water = np.maximum(water, 1 - vegetation)
    # Sun-exposed soil, paving and roofs can exceed ventilated PV temperatures.
    # Moist vegetation and water respond differently. No PV-relative remapping,
    # background mean target, or scene-wide contrast compression is applied.
    soil = 190.0 + luma * 28.0 + drift * 0.35 + broad * 1.8 + fine * 3.0 + microtexture
    # Vegetation loses solar loading through moisture/evaporation. A small
    # material-specific offset separates it from ventilated PV without a
    # scene-wide background suppression or PV-relative contrast rule.
    grass = 141.0 + luma * 49.0 + drift * 0.35 + broad * 1.5 + fine * 2.8 + microtexture * 1.15
    roof = 178.0 + luma * 39.0 + drift * 0.35 + broad * 1.4 + fine * 1.5 + microtexture * 0.35
    hard_surface = 176.0 + luma * 36.0 + drift * 0.35 + broad * 1.1 + fine * 1.4 + microtexture * 0.30
    water_temp = 79.0 + luma * 12.0 + drift * 0.48 + broad * 1.3
    if location_id in {"desert_track", "desert_farm"}:
        result = 194.0 + luma * 23.0 + drift * 0.40 + broad * 1.4 + fine * 2.4 + microtexture * 0.85
    elif location_id in {"rooftop", "rooftop_warehouse"}:
        result = roof
    else:
        result = soil * (1 - vegetation) + grass * vegetation
    if location_id not in {"rooftop", "rooftop_warehouse"}:
        # Paving retains its own material structure; it is not forced cool.
        weight = cv2.GaussianBlur(neutral_surface, (0, 0), 0.8 * scale)
        result = result * (1 - weight) + hard_surface * weight
    if material_masks is not None:
        # Soften semantic transitions in temperature, rather than painting
        # binary purple/orange regions. Canonical PV support remains separate.
        for name, field in (("vegetation", grass), ("soil", soil), ("roof_concrete", roof)):
            if name in material_masks:
                weight = cv2.GaussianBlur((material_masks[name] > 0).astype(np.float32), (0, 0), 1.1 * scale)
                result = result * (1 - weight) + field * weight
        if "pv_scene_candidate" in material_masks:
            # Unaccepted blue objects are background, not synthetic PV modules.
            candidate = cv2.GaussianBlur((material_masks["pv_scene_candidate"] > 0).astype(np.float32), (0, 0), 0.8 * scale)
            result = result * (1 - candidate) + (112 + drift * 0.5) * candidate
            water *= 1 - candidate
    # Distributed surface cooling (moisture/shade) opens orange/darker regions
    # within exposed hot land instead of shifting every ground pixel together.
    surface_cooling = np.minimum(np.maximum(-fine - 0.30, 0), 2.8) * 3.4
    result -= surface_cooling * (1 - water)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.6 * scale)
    result += np.clip(detail, -24, 24) * 0.36 * (1 - water)
    result = result * (1 - water) + water_temp * water
    shadow = np.clip((0.23 - luma) / 0.18, 0, 1)
    result = result * (1 - shadow * 0.75) + (101 + drift * 0.5) * shadow * 0.75
    # A smooth highlight shoulder preserves differences in hot terrain instead
    # of flattening every value above an upper hard clip to the same yellow.
    hot = result > 220.0
    result[hot] = 220.0 + 24.0 * np.tanh((result[hot] - 220.0) / 24.0)
    return np.maximum(result, 30).astype(np.float32)


def thermal_refine_background_materials(scalar, image, location_id, module_geometries):
    """Source-structured material refinement outside the frozen v12 PV scene.

    Keep the v12 composition, frames, shadows and optics intact. A non-PV
    collar reserves their complete finite sensor footprint, including Generated
    preview downsampling. Only the more distant substrate scalar is adjusted.
    RGB details are material/exposure cues, not measured thermal temperatures.
    """
    source = np.asarray(scalar, dtype=np.float32)
    height, width = source.shape
    ratio = min(1.0, 1600.0 / max(height, width))
    size = (max(16, int(round(width * ratio))), max(16, int(round(height * ratio))))
    rgb = np.asarray((image if image.mode == "RGB" else image.convert("RGB")).resize(size), np.uint8)
    red, green, blue = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    medium = smooth - cv2.GaussianBlur(gray, (0, 0), 14.0)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.8)
    exposure = cv2.GaussianBlur(gray, (0, 0), 18.0) - cv2.GaussianBlur(gray, (0, 0), 95.0)
    seed = seed_for(location_id, "healthy-thermal-scene")
    fine = thermal_coherent_field(gray.shape, seed_for(seed, "surface-detail"), (96, 124))
    broad = thermal_coherent_field(gray.shape, seed_for(seed, "soil-moisture"), (30, 40))
    micro = thermal_surface_microtexture(gray.shape, seed_for(seed, "surface-roughness")) * 0.85
    vegetation = np.clip((2 * green - red - blue - 8) / 65.0, 0, 1)
    cooling = np.minimum(np.maximum(-fine - 0.30, 0), 2.8) * 3.4
    # Remove part of the existing synthetic substrate grain, replacing it with
    # temperature structure aligned to real grass, crop rows and ground detail.
    land_noise = (3.0 - vegetation * 0.2) * fine + (1.0 + vegetation * 0.15) * micro - cooling
    if location_id in {"floating_water", "farm_lake"}:
        # Water has no land roughness model. Pale dry pontoons/walkways are a
        # separate material, rather than being mistaken for cold water.
        chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))
        dry = np.clip((36.0 - chroma) / 25.0, 0, 1) * np.clip((gray - 115.0) / 65.0, 0, 1)
        dry = cv2.GaussianBlur(dry, (0, 0), 0.65)
        drift = thermal_scene_drift(gray.shape, seed)
        water = 80.0 + smooth * (10.0 / 255.0) + medium * 0.08 + drift * 0.15
        structure = 145.0 + smooth * 0.11 + medium * 0.16
        target = water * (1 - dry) + structure * dry
        original = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        delta = target - original
    elif location_id in {"rooftop", "rooftop_warehouse"}:
        noise = fine * 1.5 + micro * 0.35 - cooling
        delta = -noise * 0.80 - broad * 0.65 + medium * 0.36 + detail * 0.34 + exposure * 0.20
    elif location_id in {"desert_track", "desert_farm"}:
        noise = fine * 2.4 + micro * 0.85 - cooling
        delta = -noise * 0.55 + medium * 0.38 + detail * 0.20 + exposure * 0.18
    elif location_id in {"agri_field_new", "agri_rows"}:
        delta = -land_noise * 0.70 - broad * 0.60 + medium * 0.70 + detail * 0.28 + exposure * 0.12
    elif location_id == "real_farm":
        delta = -land_noise * 0.60 - broad * 0.45 + medium * 0.55 + detail * 0.24 + exposure * 0.14
    else:
        delta = -land_noise * 0.65 - broad * 0.55 + medium * 0.52 + detail * 0.24 + exposure * 0.12
    if location_id not in {"floating_water", "farm_lake"}:
        # Retain established material temperatures, hot-ground headroom and
        # existing source shadows. No PV-relative contrast or mean-temperature
        # target is involved in this bounded structural correction.
        shadow = np.clip((0.23 - smooth / 255.0) / 0.18, 0, 1)
        delta = 22.0 * np.tanh(delta / 22.0) * (1 - shadow * 0.85)
    # Release bounded material-analysis planes before allocating native fields.
    del rgb, red, green, blue, gray, smooth, medium, detail, exposure
    del fine, broad, micro, vegetation, cooling, land_noise
    delta = cv2.resize(delta.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_CUBIC)
    support = np.zeros(source.shape, np.uint8)
    for geometry in module_geometries if module_geometries is not None else []:
        points = np.asarray(geometry, np.float32)
        if points.shape == (4, 2) and np.isfinite(points).all():
            cv2.fillConvexPoly(support, np.rint(points).astype(np.int32), 1)
    distance = cv2.distanceTransform(1 - support, cv2.DIST_L2, 3)
    pixel_scale = max(1.0, max(height, width) / 2400.0)
    guard = max(26.0, 32.0 * pixel_scale) if location_id == "real_farm" else max(28.0, max(height, width) / 48.0)
    result = source.copy()
    for start in range(0, height, 128):
        rows = slice(start, start + 128)
        blend = np.clip((distance[rows] - guard) / max(12.0, guard * 0.5), 0, 1)
        blend = blend * blend * (3.0 - 2.0 * blend)
        editable = blend > 0
        roi = result[rows]
        roi[editable] = np.clip(source[rows][editable] + delta[rows][editable] * blend[editable], 0, 255)
    return result


def thermal_pv_operating_range(location_id):
    """PV exposure/ventilation priors, independent of adjacent pixel colors.

    Vegetated sites do not heat like dry mineral ground; floating modules are
    better ventilated/cooled. These are fixed material priors, never a rule to
    keep PV above/below the measured background or to force local contrast.
    """
    return {
        "grass_open": (174, 198), "scenario_1": (174, 198),
        "agri_rows": (170, 196), "agri_field_new": (170, 196),
        "desert_farm": (168, 198), "desert_track": (168, 198),
        "rooftop": (164, 190), "rooftop_warehouse": (164, 190),
        "floating_water": (154, 180), "farm_lake": (154, 180),
    }.get(location_id, (164, 196))


def healthy_thermal_module_field(size, seed, panel_min=164, panel_max=196):
    """One uninterrupted module plane; no cell/quadrant warm templates."""
    width, height = size
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    rng = np.random.default_rng(int(seed))
    midpoint = (panel_min + panel_max) * 0.5 - 1.0 + float(np.clip(rng.normal(0, 4.8), -9.0, 9.0))
    field = np.full((height, width), midpoint, dtype=np.float32)
    # A continuously chosen gradient direction/strength and independent field
    # amplitudes give near-uniform, directional and mottled interiors, without
    # assigning modules to a finite collection of visible texture templates.
    angle = float(rng.uniform(0, 2 * np.pi))
    slope = float(rng.uniform(2.0, 17.0))
    field += slope * ((x - 0.5) * np.cos(angle) + (y - 0.5) * np.sin(angle))
    broad_gain = float(rng.uniform(1.0, 4.5))
    local_gain = float(rng.uniform(2.0, 4.5))
    field += thermal_coherent_field((height, width), seed_for(seed, "broad-flow"), (2.5, 3.3)) * broad_gain
    field += thermal_coherent_field((height, width), seed_for(seed, "local-flow"), (6, 7.5)) * local_gain
    field += thermal_coherent_field((height, width), seed_for(seed, "mid-response"), (14, 19)) * 2.8
    field += thermal_coherent_field((height, width), seed_for(seed, "micro-response"), (35, 43)) * 1.45

    # Weak directional nonuniformity is continuous, not a cell grid. Its two
    # correlation lengths are chosen independently for every physical module.
    vertical = thermal_coherent_field(
        (height, width), seed_for(seed, "vertical-response"),
        (float(rng.uniform(4, 11)), float(rng.uniform(32, 65))),
    )
    horizontal = thermal_coherent_field(
        (height, width), seed_for(seed, "horizontal-response"),
        (float(rng.uniform(30, 60)), float(rng.uniform(5, 13))),
    )
    field += vertical * float(rng.uniform(0.9, 3.0)) + horizontal * float(rng.uniform(0.3, 1.8))

    # Glass/laminate conduction limits extreme local excursions compared with
    # rough soil. Preserve all coherent fields, seeds and module offsets; use
    # a smooth heat-spreading response with no hard clamp or flat plateaus.
    excursion = max(12.0, float(panel_max - panel_min) * 0.60)
    field = midpoint + excursion * np.tanh((field - midpoint) / excursion)

    # Four unequal boundary responses, varying slowly ALONG each edge as well
    # as inward. Cooling is distance-based, not a constant purple frame.
    short = float(min(width, height))
    common_width = float(rng.uniform(0.017, 0.031)) * short
    common_strength = float(rng.uniform(21.0, 33.0))
    cooling_x = np.zeros((height, width), np.float32)
    cooling_y = np.zeros((height, width), np.float32)
    diffusion_x = np.zeros((height, width), np.float32)
    diffusion_y = np.zeros((height, width), np.float32)
    for side, distance, along in (
        (0, x * max(1, width - 1), y),
        (1, (1 - x) * max(1, width - 1), y),
        (2, y * max(1, height - 1), x),
        (3, (1 - y) * max(1, height - 1), x),
    ):
        edge_width = max(0.65, common_width * float(rng.uniform(0.75, 1.30)))
        strength = common_strength * float(rng.uniform(0.65, 1.25))
        asymmetry = 1.0 + float(rng.uniform(-0.35, 0.35)) * (along - 0.5) * 2
        phase = float(rng.uniform(0, 2 * np.pi))
        asymmetry += 0.10 * np.sin(along * float(rng.uniform(3.0, 6.0)) + phase)
        along_scale = float(rng.uniform(12, 28))
        edge_irregularity = thermal_coherent_field(
            (height, width), seed_for(seed, side, "edge-emissivity"),
            (along_scale, 3) if side < 2 else (3, along_scale),
        )
        edge_spread = thermal_coherent_field(
            (height, width), seed_for(seed, side, "edge-airflow"),
            (along_scale * 0.55, 3) if side < 2 else (3, along_scale * 0.55),
        )
        # Local thermal contrast can weaken or disappear while geometry stays
        # fixed. Width and strength change smoothly along the same boundary.
        local_strength = np.clip(0.80 + edge_irregularity * 0.38, 0.04, 1.50)
        local_width = edge_width * np.clip(0.95 + edge_spread * 0.32, 0.40, 1.85)
        response = strength * asymmetry * local_strength * np.exp(-(distance / local_width) ** 1.10)
        # A weaker, wider inward conduction shoulder joins the cool frame to
        # the glass face. Reuse the existing edge variations so no new noise,
        # layout templates or random draws disturb the approved interior.
        diffusion_width = np.maximum(local_width * 3.1, short * 0.050)
        diffusion = strength * asymmetry * local_strength * 0.28 * np.exp(
            -(distance / diffusion_width) ** 1.45
        )
        if side < 2:
            cooling_x += response
            diffusion_x += diffusion
        else:
            cooling_y += response
            diffusion_y += diffusion
    # Corner overlap depends on local heat exchange, rather than one shared
    # corner emphasis. Neither this field nor edge variation moves geometry.
    corner_mix = np.clip(
        float(rng.uniform(0.30, 0.75)) + thermal_coherent_field(
            (height, width), seed_for(seed, "corner-exchange"), (3, 4)
        ) * 0.15, 0.15, 0.90,
    )
    edge_exchange = cooling_x + cooling_y - corner_mix * np.minimum(cooling_x, cooling_y)
    field -= edge_exchange
    field -= diffusion_x + diffusion_y - corner_mix * np.minimum(diffusion_x, diffusion_y)
    # Frames/seams have a different apparent IR response from the glass face.
    # Couple through the existing irregular edge-distance fields, entirely
    # inside the physical module. Nothing is painted on surrounding ground.
    frame_apparent = float(rng.uniform(109.0, 127.0))
    frame_exchange = (1 - np.exp(-edge_exchange / 22.0)) * 0.90
    field -= np.maximum(field - frame_apparent, 0) * frame_exchange
    return field.astype(np.float32)


def thermal_elevation_response(scalar, support_mask, seed, location_id=None, geometries=None, source_image=None):
    """Geometry-projected ground shade, separate from module edge cooling.

    Only visible ground receives this response. The geometry is read-only, and
    the projected footprint is a lighting mask, never a recovered PV mask.
    Unknown mounting heights use a small dimensionless height/footprint prior;
    this is an appearance model, not a solar-position or radiometric estimate.
    """
    h, w = scalar.shape
    ratio = min(1.0, 1600.0 / max(h, w))
    size = (max(8, int(round(w * ratio))), max(8, int(round(h * ratio))))
    surface = cv2.resize(scalar, size, interpolation=cv2.INTER_AREA)
    support = cv2.resize((support_mask > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    scale = max(0.7, max(size) / 2100.0)
    outside = (1 - support).astype(np.float32)
    quads = []
    for geometry in geometries if geometries is not None else []:
        points = np.asarray(geometry, np.float32)
        if points.shape == (4, 2) and np.isfinite(points).all():
            quads.append(points * np.asarray([size[0] / w, size[1] / h], np.float32))
    if not quads:
        # Compatibility only: derive shading envelopes from existing support,
        # never change the support or rerun module recovery/detection.
        closed = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = [cv2.boxPoints(cv2.minAreaRect(c)) for c in contours if cv2.contourArea(c) > 3]
    if not quads:
        return scalar
    footprint = np.zeros(surface.shape, np.float32)
    for quad in quads:
        cv2.fillConvexPoly(footprint, np.rint(quad).astype(np.int32), 1.0)

    rgb = None
    if source_image is not None:
        rgb = np.asarray((source_image if source_image.mode == "RGB" else source_image.convert("RGB")).resize(size), np.float32)
    # One synthetic illumination direction per Generated scene, independent of
    # module rotation/selection. For Real Farm, use original RGB shade evidence
    # outside the tables, not the desired PV/background temperature contrast.
    direction = np.asarray([0.40, 1.0], np.float32)
    if location_id == "real_farm" and rgb is not None:
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) / 255.0
        residual = luma - cv2.GaussianBlur(luma, (0, 0), 12.0 * scale)
        envelope = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        reach = 7.0 * scale
        sides = []
        for dx, dy in ((-reach, 0), (reach, 0), (0, -reach), (0, reach)):
            shifted = cv2.warpAffine(
                envelope, np.asarray([[1, 0, dx], [0, 1, dy]], np.float32), size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            weight = np.maximum(shifted - envelope, 0) * outside
            sides.append(float(np.sum(residual * weight) / max(1.0, float(weight.sum()))))
        inferred = np.asarray([sides[0] - sides[1], sides[2] - sides[3]], np.float32)
        if float(np.linalg.norm(inferred)) > 0.012:
            direction = inferred
    direction /= max(1e-6, float(np.linalg.norm(direction)))

    water_site = location_id in {"floating_water", "farm_lake"}
    roof_site = location_id in {"rooftop", "rooftop_warehouse"}
    desert_site = location_id in {"desert_farm", "desert_track"}
    height_fraction = 0.035 if water_site else 0.045 if roof_site else 0.085
    projected = np.zeros(surface.shape, np.float32)
    near_edge = np.zeros(surface.shape, np.float32)
    reaches = []
    for quad in quads:
        edges = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(edges, axis=1)
        local_direction = direction.copy()
        if location_id == "real_farm":
            # The occluding edge normal comes from the actual local quad. Keep
            # the scene illumination sign while following rotated/skew tables.
            normals = np.stack((edges[:, 1], -edges[:, 0]), axis=1)
            midpoints = (quad + np.roll(quad, -1, axis=0)) * 0.5
            normals *= np.where(np.sum(normals * (midpoints - quad.mean(axis=0)), axis=1) < 0, -1, 1)[:, None]
            normals /= np.maximum(lengths[:, None], 1e-6)
            normal = normals[int(np.argmax(normals @ direction))]
            local_direction = direction * 0.82 + normal * 0.18
            local_direction /= max(1e-6, float(np.linalg.norm(local_direction)))
        # Near/far edges need different projected pixel displacements under
        # perspective. Use each vertex's adjacent physical edge spans, rather
        # than a scene-wide pixel offset or a dilated outline.
        reach = np.clip(np.minimum(lengths, np.roll(lengths, 1)) * height_fraction, 0.7 * scale, 24.0 * scale)
        displacement = reach[:, None] * local_direction[None, :]
        reaches.append(float(np.median(reach)))
        for mask, fraction in ((projected, 1.0), (near_edge, 0.28)):
            swept = cv2.convexHull(np.concatenate((quad, quad + displacement * fraction)).astype(np.float32))
            cv2.fillConvexPoly(mask, np.rint(swept).astype(np.int32), 1.0)
    # Remove the unshifted table footprint. There is no all-around contact halo.
    projected = np.maximum(projected - footprint, 0)
    near_edge = np.maximum(near_edge - footprint, 0)
    sigma = max(0.45, float(np.median(reaches)) * 0.14 + 0.35 * scale)
    if water_site:
        sigma *= 1.6
    coverage = cv2.GaussianBlur(projected, (0, 0), sigma) * 0.90
    coverage += cv2.GaussianBlur(near_edge, (0, 0), max(0.4, sigma * 0.55)) * 0.14
    irregularity = thermal_coherent_field(surface.shape, seed_for(seed, "mount-occlusion"), (110, 150))
    coverage *= np.clip(0.94 + irregularity * 0.07, 0.72, 1.10)

    # Lost solar loading depends on substrate, not its relation to PV heat.
    # Wet surfaces change less; preheated dry soil/concrete retain stored heat.
    shade_fraction = 0.50 if desert_site else 0.43 if roof_site else 0.46
    ambient = 124.0 if desert_site else 120.0 if roof_site else 106.0
    if rgb is not None and not roof_site and not desert_site:
        red, green, blue = [rgb[:, :, i] for i in range(3)]
        vegetation = np.clip((2 * green - red - blue - 8) / 65.0, 0, 1)
        water = np.clip((blue - red - 12) / 40.0, 0, 1) * np.clip((blue - green + 8) / 35.0, 0, 1)
        shade_fraction = (0.52 - vegetation * 0.14) * (1 - water) + 0.10 * water
        ambient = (108 - vegetation * 8) * (1 - water) + 77 * water
    if water_site:
        shade_fraction, ambient = 0.10, 77.0
    cooling = np.clip(coverage, 0, 1) * np.maximum(surface - ambient, 0) * shade_fraction * outside
    cooling = cv2.resize(cooling, (w, h), interpolation=cv2.INTER_LINEAR)
    result = np.empty_like(scalar, dtype=np.float32)
    for start in range(0, h, 128):
        rows = slice(start, start + 128)
        result[rows] = scalar[rows] - cooling[rows] * (support_mask[rows] == 0)
    return result


def thermal_module_frame_continuity(scalar, support_mask, geometries, seed=None):
    """Resolve all four geometry sides in scalar space before the camera PSF.

    A narrow, irregular cooling ceiling repairs weak thermal frame segments;
    stronger existing edges are retained. No brightness-derived geometry,
    union-contour inference, RGB outlines, new interior texture or sharpening.
    """
    result = np.asarray(scalar, dtype=np.float32).copy()
    height, width = result.shape
    if np.asarray(support_mask).shape != result.shape:
        raise ValueError("Physical frame support must match the healthy scalar scene.")
    pixel_scale = max(1.0, max(height, width) / 2400.0)
    for geometry in geometries if geometries is not None else []:
        points = np.asarray(geometry, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        x1, y1 = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - 1)
        x2, y2 = np.minimum([width, height], np.ceil(points.max(axis=0)).astype(int) + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        local = points - np.asarray([x1, y1], np.float32)
        shape = (int(y2 - y1), int(x2 - x1))
        yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
        distance = np.full(shape, np.inf, np.float32)
        face_reference = np.zeros(shape, np.float32)
        rim_reference = np.zeros(shape, np.float32)
        short = float(np.min(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)))
        face_inset = min(max(6.0, short * 0.10), short * 0.25)
        for a, b in zip(local, np.roll(local, -1, axis=0)):
            edge = b - a
            fraction = np.clip(((xx-a[0])*edge[0] + (yy-a[1])*edge[1]) /
                               max(1e-6, float(edge @ edge)), 0, 1)
            side_distance = np.hypot(xx-a[0]-fraction*edge[0], yy-a[1]-fraction*edge[1])
            normal = np.asarray([-edge[1], edge[0]], np.float32)
            normal /= max(1e-6, float(np.linalg.norm(normal)))
            if float(normal @ (local.mean(axis=0)-(a+b)*0.5)) < 0:
                normal = -normal
            face_x = a[0] + fraction*edge[0] + normal[0]*face_inset + x1
            face_y = a[1] + fraction*edge[1] + normal[1]*face_inset + y1
            face = cv2.remap(np.asarray(scalar, dtype=np.float32), face_x.astype(np.float32), face_y.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rim = cv2.remap(np.asarray(scalar, dtype=np.float32),
                            (a[0]+fraction*edge[0]+x1).astype(np.float32),
                            (a[1]+fraction*edge[1]+y1).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            face_reference = np.where(side_distance < distance, face, face_reference)
            rim_reference = np.where(side_distance < distance, rim, rim_reference)
            distance = np.minimum(distance, side_distance)
        inside = np.zeros(shape, np.uint8)
        raster = np.rint(points).astype(np.int32) - np.asarray([x1, y1], np.int32)
        cv2.fillConvexPoly(inside, raster, 1)
        # Physical frame rims sit between outer module polygons and the inset
        # active-glass mask. Include that real rim, without altering either
        # authoritative mask or polygon, rather than leaving a warm gap there.
        frame_seed = seed_for(tuple(np.round(points, 2).ravel()), "physical-frame-continuity")
        if seed is not None:
            frame_seed = seed_for(frame_seed, seed)
        variation = thermal_coherent_field(shape, frame_seed, (5, 7))
        width_variation = thermal_coherent_field(shape, seed_for(frame_seed, "width"), (4, 6))
        frame_width = min(short * 0.10, max(1.35 * pixel_scale, min(short * 0.025, 2.8 * pixel_scale)))
        local_width = frame_width * np.clip(1.0 + width_variation * 0.14, 0.76, 1.24)
        apparent = 115.0 + np.clip(variation * 7.0, -11.0, 11.0)
        # In cool/shaded modules the same physical rim still needs a small
        # contrast to its own adjacent face. This changes only frame response,
        # never background temperatures or the authoritative module interior.
        # Hot substrate mixed into an inset rim can otherwise erase its cool
        # response under the unchanged PSF. Bound this pre-optics compensation
        # by local scalar rim/face contrast; never sharpen the finished image.
        bleed = np.maximum(rim_reference - face_reference, 0) * 0.35
        apparent = np.maximum(42.0, np.minimum(apparent, face_reference - 18.0 - bleed))
        # The rising scalar ceiling limits the repair to a narrow smooth rim.
        # Min-composition preserves naturally stronger edges and intersections
        # without doubled cooling or uniform corner dots at shared seams.
        ceiling = apparent + 62.0 * (distance / np.maximum(0.35, local_width)) ** 1.65
        roi = result[y1:y2, x1:x2]
        roi[:] = np.where(inside > 0, np.minimum(roi, ceiling), roi)
    return result


def thermal_boundary_response(scalar, support_mask, seed, location_id=None, geometries=None, source_image=None,
                              module_geometries=None):
    """Thermal, not geometric, imperfection at the authoritative PV boundary.

    Local module temperatures influence separator contrast; a spatially varying
    scalar PSF softens some edge sections more than others. No polygons or masks
    are changed, and faults are injected only after this healthy composition.
    """
    scalar = np.asarray(scalar, dtype=np.float32)
    inside = (np.asarray(support_mask) > 0).astype(np.uint8)
    if not np.any(inside):
        return scalar
    h, w = scalar.shape
    scale = max(0.8, max(h, w) / 2100.0)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    distance += cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
    fine = thermal_coherent_field(scalar.shape, seed_for(seed, "boundary-local-transfer"), (110, 150))
    # The distance plane becomes the band; row expressions preserve float32
    # operation order without allocating several more native-size planes.
    band = distance
    for start in range(0, h, 128):
        rows = slice(start, start + 128)
        band[rows] = np.exp(-band[rows] / (1.7 * scale * np.clip(1 + fine[rows] * 0.25, 0.55, 1.65)))
    uneven = thermal_coherent_field(scalar.shape, seed_for(seed, "boundary-transfer"), (65, 90))
    pv_weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 2.2 * scale)
    out = np.empty_like(scalar)
    np.multiply(scalar, inside, out=out)
    cv2.GaussianBlur(out, (0, 0), 2.2 * scale, dst=out)
    for start in range(0, h, 128):
        rows = slice(start, start + 128)
        pv_average = out[rows] / np.maximum(pv_weight[rows], 0.02)
        relaxation = np.clip(0.22 + uneven[rows] * 0.16 + fine[rows] * 0.12, 0.0, 0.55) * band[rows] * np.clip(pv_weight[rows] * 3, 0, 1)
        out[rows] = scalar[rows] + np.clip(pv_average - scalar[rows], 0, 48) * relaxation
        band[rows] *= np.clip(0.30 + uneven[rows] * 0.09 + fine[rows] * 0.13, 0.04, 0.65)
    del distance, uneven, fine, pv_weight, pv_average, relaxation, inside
    softened = cv2.GaussianBlur(out, (0, 0), 1.05 * scale)
    for start in range(0, h, 128):
        rows = slice(start, start + 128)
        mix = band[rows]
        out[rows] = out[rows] * (1 - mix) + softened[rows] * mix
    del softened, band, mix
    healthy = thermal_elevation_response(out, support_mask, seed, location_id, geometries, source_image)
    del out
    return thermal_module_frame_continuity(
        healthy, support_mask, module_geometries if module_geometries is not None else geometries
    )


def thermal_context_on_module(scene_drift, points, size):
    """Sample global/row drift through the module homography, preserving locality."""
    width, height = size
    destination = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(points, dtype=np.float32), destination)
    return cv2.warpPerspective(scene_drift, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def composite_healthy_thermal_module(temp, points, scene_drift, seed, panel_min, panel_max, support_mask=None):
    """Project one healthy plane into a scene ROI; masks/geometry stay unchanged."""
    points = np.asarray(points, dtype=np.float32)
    height, width = temp.shape
    x1, y1 = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - 1)
    x2, y2 = np.minimum([width, height], np.ceil(points.max(axis=0)).astype(int) + 2)
    if x2 <= x1 or y2 <= y1:
        return
    local_width = max(24, min(480, int(np.linalg.norm(points[1] - points[0])) + 1))
    local_height = max(24, min(720, int(np.linalg.norm(points[3] - points[0])) + 1))
    size = (local_width, local_height)
    local = healthy_thermal_module_field(size, seed, panel_min, panel_max)
    local += thermal_context_on_module(scene_drift, points, size)
    canonical = np.asarray([[0, 0], [local_width - 1, 0], [local_width - 1, local_height - 1], [0, local_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(canonical, points - np.asarray([x1, y1], dtype=np.float32))
    roi_size = (int(x2 - x1), int(y2 - y1))
    # Replicate the cooled edge rather than interpolating scalar values with zero.
    warped = cv2.warpPerspective(local, matrix, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    support = cv2.warpPerspective(np.ones((local_height, local_width), np.float32), matrix, roi_size, flags=cv2.INTER_LINEAR)
    if support_mask is not None:
        support *= support_mask[y1:y2, x1:x2]
    roi = temp[y1:y2, x1:x2]
    roi[:] = roi * (1 - support) + warped * support
