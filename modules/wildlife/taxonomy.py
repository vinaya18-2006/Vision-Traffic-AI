"""Validated, versioned mapping from detector labels to dietary categories."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .errors import TaxonomyConfigError

DietaryCategory = Literal["carnivore", "non_carnivore", "unknown"]
VALID_CATEGORIES = frozenset({"carnivore", "non_carnivore", "unknown"})
DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")


def default_taxonomy_path() -> Path:
    """Return the configured taxonomy JSON path without silently substituting one."""
    configured = os.environ.get("WILDLIFE_TAXONOMY_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_TAXONOMY_PATH


@dataclass(frozen=True)
class WildlifeTaxonomy:
    """Canonical supported detector labels and explicit diet assignments."""

    version: str
    source: str
    animal_classes: tuple[str, ...]
    classifications: Mapping[str, DietaryCategory]
    notes: Mapping[str, str]
    references: tuple[Mapping[str, str], ...]

    def classification_for(self, species: str) -> DietaryCategory:
        """Return the configured dietary category, defaulting to not classified."""
        return self.classifications.get(_normalize(species), "unknown")

    def supports(self, class_name: str) -> bool:
        """Return whether a model label is configured as an animal class."""
        return _normalize(class_name) in self.animal_classes


def load_taxonomy(path: Path | str | None = None) -> WildlifeTaxonomy:
    """Load and validate the dedicated wildlife taxonomy JSON file."""
    taxonomy_path = Path(path).expanduser().resolve() if path is not None else default_taxonomy_path()
    try:
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaxonomyConfigError(f"Could not read wildlife taxonomy '{taxonomy_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyConfigError(f"Wildlife taxonomy '{taxonomy_path}' is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise TaxonomyConfigError("Wildlife taxonomy must contain a JSON object.")
    version = data.get("version")
    source = data.get("source")
    raw_classes = data.get("animal_classes")
    raw_classifications = data.get("classifications")
    raw_notes = data.get("notes", {})
    raw_references = data.get("references", [])
    if not isinstance(version, str) or not version.strip():
        raise TaxonomyConfigError("Wildlife taxonomy must declare a non-empty version.")
    if not isinstance(source, str) or not source.strip():
        raise TaxonomyConfigError("Wildlife taxonomy must declare a mapping source.")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise TaxonomyConfigError("Wildlife taxonomy animal_classes must be a non-empty list.")
    if not isinstance(raw_classifications, dict):
        raise TaxonomyConfigError("Wildlife taxonomy classifications must be an object.")

    classes = tuple(_normalize(item) for item in raw_classes if isinstance(item, str) and item.strip())
    if len(classes) != len(raw_classes) or len(set(classes)) != len(classes):
        raise TaxonomyConfigError("Wildlife animal class labels must be non-empty and unique.")

    classifications: dict[str, DietaryCategory] = {}
    for label, category in raw_classifications.items():
        if not isinstance(label, str) or not label.strip():
            raise TaxonomyConfigError("Taxonomy classification labels must be non-empty strings.")
        normalized_label = _normalize(label)
        if normalized_label not in classes:
            raise TaxonomyConfigError(
                f"Taxonomy mapping '{label}' is not listed in animal_classes."
            )
        if normalized_label in classifications:
            raise TaxonomyConfigError(
                f"Taxonomy class '{normalized_label}' is mapped more than once."
            )
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise TaxonomyConfigError(
                f"Taxonomy category for '{label}' must be one of "
                f"{', '.join(sorted(VALID_CATEGORIES))}."
            )
        classifications[normalized_label] = category
    if set(classifications) != set(classes):
        missing = ", ".join(sorted(set(classes) - set(classifications)))
        raise TaxonomyConfigError(f"Every supported animal class needs an explicit mapping: {missing}.")

    if not isinstance(raw_notes, dict) or not all(
        isinstance(label, str) and isinstance(note, str)
        for label, note in raw_notes.items()
    ):
        raise TaxonomyConfigError("Wildlife taxonomy notes must map class names to text.")
    if not isinstance(raw_references, list) or not all(
        isinstance(reference, dict)
        and isinstance(reference.get("title"), str)
        and isinstance(reference.get("url"), str)
        for reference in raw_references
    ):
        raise TaxonomyConfigError("Wildlife taxonomy references must include title and URL strings.")

    return WildlifeTaxonomy(
        version=version.strip(),
        source=source.strip(),
        animal_classes=classes,
        classifications=classifications,
        notes={_normalize(label): note for label, note in raw_notes.items()},
        references=tuple(raw_references),
    )


def _normalize(label: str) -> str:
    return label.strip().casefold()
