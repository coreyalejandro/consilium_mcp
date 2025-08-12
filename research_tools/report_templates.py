"""
Structured report templates and helpers.

Controlled via env var STRUCTURED_REPORT_MODE=1 and STRUCTURED_REPORT_TYPE.
Default type: enhancement_plan_v1
"""

from __future__ import annotations

from typing import Dict


def get_report_template(report_type: str = "enhancement_plan_v1") -> str:
    if report_type == "enhancement_plan_v1":
        return (
            "# Enhancement Plan\n\n"
            "## Executive Summary\n"
            "- Objective\n"
            "- Approach\n"
            "- Key Results\n\n"
            "## Top 10 Enhancements (Ranked)\n"
            "| Rank | Enhancement | Value Rationale | Effort | Risk |\n"
            "|---:|---|---|---:|---|\n"
            "| 1 |  |  |  |  |\n"
            "| 2 |  |  |  |  |\n"
            "| 3 |  |  |  |  |\n"
            "| 4 |  |  |  |  |\n"
            "| 5 |  |  |  |  |\n"
            "| 6 |  |  |  |  |\n"
            "| 7 |  |  |  |  |\n"
            "| 8 |  |  |  |  |\n"
            "| 9 |  |  |  |  |\n"
            "| 10 |  |  |  |  |\n\n"
            "## Detailed Rationale\n"
            "- Why these enhancements\n"
            "- Impact analysis\n"
            "- Dependencies\n\n"
            "## Implementation Plan (Code Included)\n"
            "### Overview\n"
            "- Milestones\n"
            "- Timeline\n\n"
            "### Edits\n"
            "Provide code edits for each enhancement. Use fenced code blocks.\n\n"
            "## Risks & Mitigations\n"
            "- Risk\n"
            "- Mitigation\n\n"
            "## Next Steps\n"
            "- Immediate actions\n"
            "- Owners\n"
        )
    return "# Report\n\n## Summary\n\n## Details\n"


