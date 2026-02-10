import os
import argparse

def load_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', default='pairs_CALFW.txt', help='Path to pairs_CALFW.txt')
    parser.add_argument('--output_dir', default='.', help='Directory to save output CSVs')
    args = parser.parse_args()

    if not os.path.exists(args.source_file):
        print(f"Error: Source file '{args.source_file}' not found.")
        return

    lines = load_file(args.source_file)
    print(f"Total lines: {len(lines)}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Genuine
    with open(os.path.join(args.output_dir, 'calfw_genuine.csv'), 'w') as f:
        f.write('probe_fn,probe_label,gallery_fn,gallery_label,genuine\n')
        
        parsed_line = ""
        for i, line in enumerate(lines[:6000]):
            splited_line = line.strip().replace('\n', '').split(" ")
            parsed_line += f"{splited_line[0]},{i},"
            if (i + 1) % 2 == 0:
                parsed_line += f"{1}\n"
                f.write(parsed_line)
                parsed_line = ""

    # Impostor
    with open(os.path.join(args.output_dir, 'calfw_impostor.csv'), 'w') as f:
        f.write('probe_fn,probe_label,gallery_fn,gallery_label,genuine\n')
        
        parsed_line = ""
        for i, line in enumerate(lines[6000:]):
            splited_line = line.strip().replace('\n', '').split(" ")
            parsed_line += f"{splited_line[0]},{i},"
            if (i + 1) % 2 == 0:
                parsed_line += f"{0}\n"
                f.write(parsed_line)
                parsed_line = ""

if __name__ == "__main__":
    main()
