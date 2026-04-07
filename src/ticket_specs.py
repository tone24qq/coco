from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

SizeClass = Literal["20", "80", "120"]


@dataclass(frozen=True)
class TicketSpec:
    size_class: str
    expected_rows: int
    expected_cols: int
    page_count: int
    page_merge_axis: str
    user_index_base: int
    allowed_mask_labels: tuple[str, ...]

    @property
    def expected_shape(self) -> tuple[int, int]:
        return self.expected_rows, self.expected_cols

    @property
    def legal_values(self) -> set[int]:
        return set(range(1, self.expected_rows * self.expected_cols + 1))


TICKET_SPECS: Dict[str, TicketSpec] = {
    "20": TicketSpec("20", 5, 4, 1, "none", 1, ("solid_black",)),
    "80": TicketSpec("80", 10, 8, 1, "none", 1, ("solid_black",)),
    "120": TicketSpec("120", 12, 10, 2, "vertical", 1, ("solid_black",)),
}


class TicketSpecError(ValueError):
    pass


def build_ticket_spec(
    rows: int,
    cols: int,
    *,
    page_count: int = 1,
    page_merge_axis: str = "none",
    size_class: str = "generic",
) -> TicketSpec:
    if rows <= 0 or cols <= 0:
        raise TicketSpecError("invalid_shape")
    return TicketSpec(
        size_class=size_class,
        expected_rows=int(rows),
        expected_cols=int(cols),
        page_count=int(page_count),
        page_merge_axis=page_merge_axis,
        user_index_base=1,
        allowed_mask_labels=("solid_black",),
    )


def get_ticket_spec(size_class: str) -> TicketSpec:
    if size_class not in TICKET_SPECS:
        raise TicketSpecError(f"unknown_size_class:{size_class}")
    return TICKET_SPECS[size_class]


def get_ticket_spec_by_shape(rows: int, cols: int) -> TicketSpec:
    for spec in TICKET_SPECS.values():
        if spec.expected_shape == (rows, cols):
            return spec
    return build_ticket_spec(rows, cols)


def detect_size_class_from_path(image_path: str) -> Optional[SizeClass]:
    p = Path(image_path)
    for part in p.parts:
        if part in TICKET_SPECS:
            return part  # type: ignore[return-value]
    return None


def resolve_ticket_spec(
    *,
    size_class: str | None = None,
    rows: int | None = None,
    cols: int | None = None,
    image_path: str | None = None,
) -> TicketSpec:
    if rows is not None or cols is not None:
        if rows is None or cols is None:
            raise TicketSpecError("rows_cols_must_be_paired")
        return build_ticket_spec(rows, cols)
    if size_class is not None:
        return get_ticket_spec(size_class)
    if image_path is not None:
        inferred = detect_size_class_from_path(image_path)
        if inferred is not None:
            return get_ticket_spec(inferred)
    raise TicketSpecError("shape_resolution_failed")


def validate_page_contract(spec: TicketSpec, image_paths: List[str]) -> None:
    if len(image_paths) != spec.page_count:
        raise TicketSpecError("page_merge_invalid")
    if spec.page_count == 2:
        names = [Path(x).stem for x in image_paths]
        if not ("頁面_1" in names[0] and "頁面_2" in names[1]):
            raise TicketSpecError("page_merge_invalid")
