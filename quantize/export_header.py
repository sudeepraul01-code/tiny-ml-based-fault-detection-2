"""Export a TFLite FlatBuffer using a TFLite Micro-friendly C array layout."""

from pathlib import Path


def render_c_header(model_bytes: bytes, array_name: str = "fault_model_int8") -> str:
    """Format a model FlatBuffer as a self-contained C header."""
    if not model_bytes:
        raise ValueError("model_bytes must not be empty")
    if not array_name.isidentifier():
        raise ValueError("array_name must be a valid C identifier")
    lines = [
        "#ifndef TINYML_FAULT_MODEL_DATA_H_",
        "#define TINYML_FAULT_MODEL_DATA_H_",
        "",
        "#include <stdint.h>",
        "",
        f"alignas(16) const unsigned char {array_name}[] = {{",
    ]
    for start in range(0, len(model_bytes), 12):
        chunk = model_bytes[start : start + 12]
        lines.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int {array_name}_len = {len(model_bytes)};",
            "",
            "#endif  // TINYML_FAULT_MODEL_DATA_H_",
            "",
        ]
    )
    return "\n".join(lines)


def write_c_header(model_bytes: bytes, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_c_header(model_bytes), encoding="ascii")
    return path
