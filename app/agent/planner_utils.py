import re
from typing import Any, Dict, List, Optional, Set, Tuple


def build_page_filter(full_filename: str, page_num: int) -> Dict[str, Any]:
    """Return a Chroma-compatible filter for a specific PDF page."""
    return {
        "$and": [
            {"file_name": {"$eq": full_filename}},
            {"page": {"$eq": page_num}},
        ]
    }


def image_has_content(img_info: Dict[str, Any]) -> bool:
    """Determine whether an image metadata entry contains meaningful information."""
    if not img_info:
        return False

    ocr_text = str(img_info.get("ocr_text", "") or "").strip()
    visual_desc = str(img_info.get("visual_description", "") or "").strip()
    caption = str(img_info.get("caption", "") or "").strip()
    figure_number = str(img_info.get("figure_number", "") or "").strip()
    captions = img_info.get("captions", []) or []

    if ocr_text or visual_desc or caption or figure_number:
        return True

    return any(str(item or "").strip() for item in captions)


def normalize_page_image_captions(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each image on a page is paired with the most relevant caption detail.

    Vector-store metadata often attaches the same caption array to every image on a page.
    This helper distributes caption details across images so the UI can render distinct
    figure numbers and descriptions.
    """
    if not images:
        return images

    grouped_images: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for img in images:
        filename = (img.get("filename") or "").lower()
        page = img.get("page")
        key: Tuple[str, Any] = (filename, img.get("image_path")) if page is None else (filename, page)
        grouped_images.setdefault(key, []).append(img)

    for group in grouped_images.values():
        if len(group) <= 1:
            continue

        caption_pool: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        used_indices: Set[int] = set()

        for img in group:
            details = img.get("caption_details") or []
            if not isinstance(details, list):
                continue
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                figure_number = (detail.get("figure_number") or "").strip()
                caption_text = (detail.get("caption_text") or "").strip()
                if not figure_number and not caption_text:
                    continue
                identifier = (figure_number.lower(), caption_text.lower())
                if identifier in seen:
                    continue
                seen.add(identifier)
                caption_pool.append(detail)

        if not caption_pool:
            continue

        def take_detail_by_figure(fig: Optional[str]) -> Optional[Dict[str, Any]]:
            if not fig:
                return None
            normalized_fig = fig.strip().lower()
            for idx, detail in enumerate(caption_pool):
                if idx in used_indices:
                    continue
                if (detail.get("figure_number") or "").strip().lower() == normalized_fig:
                    used_indices.add(idx)
                    return detail
            return None

        def take_next_detail(start_index: int = 0) -> Dict[str, Any]:
            for idx in range(start_index, len(caption_pool)):
                if idx not in used_indices:
                    used_indices.add(idx)
                    return caption_pool[idx]
            for idx, detail in enumerate(caption_pool):
                if idx not in used_indices:
                    used_indices.add(idx)
                    return detail
            return caption_pool[-1]

        def apply_detail(target: Dict[str, Any], detail: Dict[str, Any]) -> None:
            if not isinstance(detail, dict):
                return

            figure_number = detail.get("figure_number")
            caption_text = detail.get("caption_text") or detail.get("full_caption")

            if figure_number:
                target["figure_number"] = figure_number

            if caption_text:
                target["caption"] = caption_text
                target["captions"] = [caption_text]
                current_desc = str(target.get("visual_description", "") or "").strip()
                cleaned_description = _clean_visual_description(current_desc, caption_text, figure_number)
                if cleaned_description.strip().lower() == caption_text.strip().lower():
                    # Avoid duplicating the caption verbatim in the image description.
                    cleaned_description = ""
                target["visual_description"] = cleaned_description

            target["caption_details"] = [detail]

        figures = [
            (img.get("figure_number") or "").strip().lower()
            for img in group
            if img.get("figure_number")
        ]
        if len(figures) == len(group) and len(set(figures)) == len(group):
            for img in group:
                matched = take_detail_by_figure((img.get("figure_number") or "").strip().lower())
                if not matched:
                    matched = take_next_detail()
                apply_detail(img, matched)
            continue

        for index, img in enumerate(group):
            matched = take_detail_by_figure((img.get("figure_number") or "").strip().lower())
            if not matched:
                matched = take_next_detail(index)
            apply_detail(img, matched)

    return images


def _clean_visual_description(
    current_description: str,
    caption_text: str,
    figure_number: Optional[str],
) -> str:
    """
    Ensure the visual description complements (not duplicates) the figure caption.

    Removes leading "Figure NN" prefixes, trims whitespace, and falls back to the
    caption text when the existing description is empty or generic.
    """
    description = current_description.strip()
    if not description or description.lower() in {"", "wide diagram or chart"}:
        return ""

    lower_desc = description.lower()
    prefix = ""
    if figure_number:
        prefix = f"figure {figure_number}".lower()

    # If description references a different figure number, drop it.
    if figure_number:
        match = re.search(r"figure\s+(\d+(?:\.\d+)?)", lower_desc)
        if match:
            matched_number = match.group(1).strip().lower()
            if matched_number != figure_number.strip().lower():
                return ""

    if prefix and lower_desc.startswith(prefix):
        # Remove the leading figure label and any separators immediately after it.
        trimmed = description[len(prefix):].lstrip(": .-").strip()
        description = trimmed

    # If description still mirrors the caption exactly, shorten it to avoid duplication.
    if caption_text and description.strip().lower() == caption_text.strip().lower():
        return ""

    return description

