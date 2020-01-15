import os

import strip_hints


def clean_up_type_hints(s: str) -> str:
    lines = s.split(os.linesep)
    lines = [
        line
        for line in lines
        if (not line.startswith("from typing") and not line.strip().startswith("#"))
        or line.strip().startswith("# coding")
    ]
    return "\n".join(lines)


def main() -> None:
    input_dir = os.path.join("..", "plugins")
    output_dir = os.path.join("..", "plugins_py36")

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        source = strip_hints.strip_file_to_string(input_file)
        source = clean_up_type_hints(source)

        output_file = os.path.join(output_dir, file)
        with open(output_file, "w", encoding="utf8") as file_obj:
            file_obj.write(source)


if __name__ == "__main__":
    main()
