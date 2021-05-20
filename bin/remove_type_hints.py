import subprocess
from pathlib import Path

import strip_hints


def clean_up_type_hints(s: str) -> str:
    lines = s.split("\n")
    lines = [line for line in lines if (not line.startswith("from typing "))]
    return "\n".join(lines)


def run() -> None:
    input_dir = Path("../plugins")
    output_dir = Path("../plugins_py36")

    output_dir.mkdir(exist_ok=True, parents=True)
    output_files = []

    for input_file in input_dir.iterdir():
        source = strip_hints.strip_file_to_string(input_file)
        source = clean_up_type_hints(source)

        output_file = output_dir / input_file.name
        with output_file.open("w", encoding="utf8") as file_obj:
            file_obj.write(source)

        output_files.append(output_file)

    subprocess.run(["python", "-m", "black"] + [str(f) for f in output_files])


if __name__ == "__main__":
    run()
