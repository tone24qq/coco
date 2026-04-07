from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

SizeClass = Literal["20", "80", "120"]


@dataclass(frozen=True)
class TicketSpec:
    size_class: SizeClass
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
    "20": TicketSpec(
        "20",
        expected_rows=5,
        expected_cols=4,
        page_count=1,
        page_merge_axis="none",
        user_index_base=1,
        allowed_mask_labels=("solid_black",),
    ),
    "80": TicketSpec(
        "80",
        expected_rows=10,
        expected_cols=8,
        page_count=1,
        page_merge_axis="none",
        user_index_base=1,
        allowed_mask_labels=("solid_black",),
    ),
    "120": TicketSpec(
        "120",
        expected_rows=12,
        expected_cols=10,
        page_count=2,
        page_merge_axis="vertical",
        user_index_base=1,
        allowed_mask_labels=("solid_black",),
    ),
}


class TicketSpecError(ValueError):
    pass


def get_ticket_spec(size_class: str) -> TicketSpec:
    if size_class not in TICKET_SPECS:
        raise TicketSpecError(f"unknown_size_class:{size_class}")
    return TICKET_SPECS[size_class]


def detect_size_class_from_path(image_path: str) -> SizeClass:
    p = Path(image_path)
    for part in p.parts:
        if part in TICKET_SPECS:
            return part  # type: ignore[return-value]
    raise TicketSpecError(f"size_class_not_in_path:{image_path}")


def validate_page_contract(size_class: str, image_paths: List[str]) -> None:
    spec = get_ticket_spec(size_class)
    if len(image_paths) != spec.page_count:
        raise TicketSpecError("page_merge_invalid")
    if spec.page_count == 2:
        names = [Path(x).stem for x in image_paths]
        if not ("頁面_1" in names[0] and "頁面_2" in names[1]):
            raise TicketSpecError("page_merge_invalid")
