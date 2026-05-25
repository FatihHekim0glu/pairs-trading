"""MkDocs macros hook.

Exposes a small set of project-level constants to Markdown via the
``mkdocs-macros-plugin`` interface. The hook is intentionally minimal -- the
documentation agent owns the page content -- but registers placeholders here so
they can be referenced from Markdown without rebuilding the navigation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def define_env(env: Any) -> None:
    """Register project-wide variables and macros for MkDocs templates.

    Parameters
    ----------
    env : Any
        The macros plugin environment. Typed as ``Any`` because the plugin is
        optional and may not be installed in every build.
    """
    variables: Mapping[str, str] = {
        "project_name": "pairs-trading",
        "repo_url": "https://github.com/fatihhekimoglu/pairs-trading",
    }
    for key, value in variables.items():
        env.variables[key] = value

    @env.macro
    def github_link(path: str) -> str:
        """Return a Markdown link to ``path`` in the project repository."""
        return f"[{path}]({variables['repo_url']}/blob/main/{path})"
