import json

def compare_json_files(file1, file2):
    """Compare two JSON files to check if they are identical."""
    try:
        # Load JSON content from both files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        
        # Compare the parsed JSON data
        if data1 == data2:
            print("JSON files are identical.")
            return True
        else:
            print("JSON files are not identical.")
            print("Differences found:")
            print(f"File 1 content: {data1}")
            print(f"File 2 content: {data2}")
            return False

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False


def compare_txt_files(file1, file2):
    """Compare two text files line by line to check if they are identical."""
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                if line1 != line2:
                    print("Files are not identical.")
                    print(f"Difference found:\nFile 1: {line1.strip()}\nFile 2: {line2.strip()}")
                    return False

            # Check if one file has extra lines
            extra_lines_f1 = f1.read().strip()
            extra_lines_f2 = f2.read().strip()

            if extra_lines_f1 or extra_lines_f2:
                print("Files are not identical.")
                if extra_lines_f1:
                    print(f"Extra lines in {file1}:")
                    print(extra_lines_f1)
                if extra_lines_f2:
                    print(f"Extra lines in {file2}:")
                    print(extra_lines_f2)
                return False

        print("Files are identical.")
        return True

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False


file1 = "../gns-reptile/cbgeo/datasets/indiv_traj/30_deg_0/mesh.txt"
file2 = "../30_deg_0/mesh.txt"

compare_txt_files(file1, file2)