import os

def clean_text_file(input_file: str, output_file: str):
    """
    Remove unnecessary empty lines and trailing spaces from a text file
    and save the cleaned text.
    """
    # open with utf-8-sig to handle BOM; replace undecodable bytes if any
    with open(input_file, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = f.readlines()

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        stripped = line.rstrip()  # remove trailing spaces and newline
        if stripped == "":
            # keep only a single blank line between paragraphs
            if not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
        else:
            cleaned_lines.append(stripped)
            previous_blank = False

    # make sure destination folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines))


def batch_clean(raw_folder: str = "Raw", data_folder: str = "Data"):
    """
    Clean all .txt files from raw_folder and save in data_folder.
    """
    os.makedirs(data_folder, exist_ok=True)

    for filename in os.listdir(raw_folder):
        if filename.lower().endswith(".txt"):
            in_path = os.path.join(raw_folder, filename)
            out_path = os.path.join(data_folder, filename)
            clean_text_file(in_path, out_path)
            print(f"✅ Cleaned: {filename} -> {out_path}")


if __name__ == "__main__":
    # Change folder names if needed
    batch_clean("prashant", "Data")
    print("🎉 All text files cleaned and saved to 'Data' folder.")
