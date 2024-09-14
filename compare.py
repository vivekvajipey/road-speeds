import os
import filecmp
import os
import filecmp

def compare_files(file1, file2):
    """
    Compares two files by size and content, reports any differences.

    Args:
    file1: Path to the first file.
    file2: Path to the second file.

    Returns:
    True if both files have the same size and content, False otherwise.
    """
    identical = True
    
    # Compare file sizes first
    size1, size2 = os.path.getsize(file1), os.path.getsize(file2)
    if size1 != size2:
        print(f"File sizes differ: {file1} ({size1} bytes), {file2} ({size2} bytes)")
        identical = False
    
    # If sizes match, compare the content
    if not filecmp.cmp(file1, file2, shallow=False):
        print(f"Files differ in content: {file1} and {file2}")
        identical = False
    
    return identical

def compare_directories(dir1, dir2):
    """
    Compares two directories recursively and reports all discrepancies.

    Args:
    dir1: Path to the first directory.
    dir2: Path to the second directory.

    Returns:
    True if directories are identical, False otherwise.
    """
    identical = True

    # Check if both paths are directories
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print(f"One or both paths are not directories: {dir1}, {dir2}")
        return False

    # Compare the directory trees (recursively compares the files and subdirectories)
    comparison = filecmp.dircmp(dir1, dir2)

    # Check for files/directories that exist in one but not the other
    if comparison.left_only:
        print(f"Only in {dir1}: {comparison.left_only}")
        identical = False

    if comparison.right_only:
        print(f"Only in {dir2}: {comparison.right_only}")
        identical = False

    # Compare files that exist in both directories
    for common_file in comparison.common_files:
        file1 = os.path.join(dir1, common_file)
        file2 = os.path.join(dir2, common_file)
        if not compare_files(file1, file2):
            identical = False

    # Compare subdirectories recursively
    for subdir in comparison.common_dirs:
        new_dir1 = os.path.join(dir1, subdir)
        new_dir2 = os.path.join(dir2, subdir)
        if not compare_directories(new_dir1, new_dir2):
            identical = False

    return identical

def compare_tuples_of_directories(dir_tuples):
    """
    Compares multiple pairs of directories provided as tuples.

    Args:
    dir_tuples: A list of tuples, where each tuple contains two directory paths.
    """
    for dir1, dir2 in dir_tuples:
        print(f"Comparing {dir1} with {dir2}...")
        result = compare_directories(dir1, dir2)
        if result:
            print(f"Directories {dir1} and {dir2} are identical.\n")
        else:
            print(f"Directories {dir1} and {dir2} have discrepancies.\n")


# Example usage
dir_tuples = [
    # ('/home/jeffrey/repos/road-speeds/data/rgb_crops_mo_4', '/home/jeffrey/repos/road-speeds/data2/rgb_crops_mo_4'),
    # ('/home/jeffrey/repos/road-speeds/data/nir_crops_mo_4', '/home/jeffrey/repos/road-speeds/data2/nir_crops_mo_4')
    ('/home/jeffrey/repos/road-speeds/data', '/home/jeffrey/repos/road-speeds/data2')
   
]

compare_tuples_of_directories(dir_tuples)
