"""Bounded, local diagnostics for allocation failures; never retains array data."""
import os
import traceback
from datetime import datetime, timezone

import cv2
import numpy as np


def is_allocation_failure(exc):
    return isinstance(exc, MemoryError) or (
        isinstance(exc, cv2.error) and (
            getattr(exc, "code", None) == -4
            or any(word in str(exc).lower() for word in ("outofmemory", "failed to allocate", "insufficient memory"))
        )
    )


def memory_snapshot():
    result = {"pid": os.getpid()}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Status(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_commit_limit",
                    "available_commit", "total_virtual", "available_virtual", "extended_virtual")
            ]

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "private_commit", "peak_private_commit")
            ]

        status, counters = Status(), Counters()
        status.length, counters.cb = ctypes.sizeof(status), ctypes.sizeof(counters)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result.update({name + "_bytes": int(getattr(status, name)) for name in (
                "total_physical", "available_physical", "total_commit_limit", "available_commit")})
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            result.update({name + "_bytes": int(getattr(counters, name)) for name in (
                "rss", "peak_rss", "private_commit", "peak_private_commit")})
    else:
        try:
            import resource
            result["peak_rss_platform_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
    return result


def array_inventory(values):
    """Metadata only, with shared NumPy storage counted once and bounded walk."""
    from collections import deque
    seen, storage, records = set(), set(), []
    pending = deque([(values, "root", 5)])
    visited = 0
    while pending and visited < 20000:
        value, path, depth = pending.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        visited += 1
        if isinstance(value, np.ndarray):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if id(owner) not in storage:
                storage.add(id(owner))
                records.append({"path": path, "shape": list(value.shape), "dtype": str(value.dtype),
                                "bytes": int(owner.nbytes)})
        elif depth > 0 and isinstance(value, dict):
            for key, item in value.items():
                if len(pending) < 20000:
                    pending.append((item, path + "." + str(key), depth - 1))
        elif depth > 0 and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if len(pending) < 20000:
                    pending.append((item, path + "[" + str(index) + "]", depth - 1))
        elif hasattr(value, "_real_farm_rgb_pixels"):
            pending.append((value._real_farm_rgb_pixels, path + ".rgb_cache", 1))
    return {"unique_numpy_bytes": sum(item["bytes"] for item in records),
            "largest_arrays": sorted(records, key=lambda item: -item["bytes"])[:30],
            "walk_truncated": bool(pending)}


def allocation_failure_report(exc, image=None, session=None):
    frames, local_arrays = [], {}
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        frames.append({"file": frame.f_code.co_filename, "function": frame.f_code.co_name,
                       "line": tb.tb_lineno})
        for name, value in frame.f_locals.items():
            if isinstance(value, np.ndarray):
                local_arrays[frame.f_code.co_name + "." + name] = value
        candidate_image = frame.f_locals.get("image")
        if isinstance(getattr(candidate_image, "size", None), tuple):
            image = candidate_image
        tb = tb.tb_next
    report = {"error": str(exc), "memory": memory_snapshot(), "python_frames": frames,
              "captured_at": datetime.now(timezone.utc).isoformat(), "opencv_version": cv2.__version__,
              "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
              "source_size": list(image.size) if image is not None else None,
              "active_arrays": array_inventory(local_arrays),
              "session_arrays": array_inventory(dict(session)) if session is not None else {}}
    if session is not None:
        report["source_path"] = session.get("real_farm_image_path")
        state = session.get("real_farm_state") or {}
        report["source_rgb_sha256_prefix"] = state.get("source_rgb_sha256_prefix")
    if frames:
        report["operation"] = frames[-1]
    return report
