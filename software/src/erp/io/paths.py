from pathlib import Path
from typing import Union

def get_project_root(starting_path: Path = Path(__file__).resolve()) -> Path:
    """Searches upwards for the project root marked by pyproject.toml."""
    for parent in [starting_path, *starting_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (pyproject.toml not found).")

def resolve_model_path(filename: Union[str, Path], subfolder: str = "assets") -> Path:
    """Resolves absolute path to mujoco XML assets."""
    project_root = get_project_root()
    model_path = project_root / "notebooks" / subfolder / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Return the Path object directly instead of a string
    return model_path