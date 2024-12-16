## Make manually sorting images more game-like for our own sanity
import os
import shutil
import cv2


def select_directory(prompt):
    """Prompt the user to input a directory path."""
    folder = input(f"{prompt}: ").strip()
    if os.path.isdir(folder):
        return folder
    else:
        print("Invalid directory. Please try again.")
        return select_directory(prompt)


def move_file(file_path, dest_folder):
    """Move the file to the destination folder."""
    shutil.move(file_path, os.path.join(dest_folder, os.path.basename(file_path)))


def sort_images():
    """Sort images into pre-existing folders."""
    input_folder = select_directory("Enter the path to the folder containing images")
    output_folder = select_directory("Enter output directory")
    screw_ups_list = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
    if not image_files:
        print("No PNG files found in the selected folder.")
        return

    for image_file in image_files:
        file_path = os.path.join(input_folder, image_file)

        # Display the image using OpenCV
        image = cv2.imread(file_path)
        cv2.imshow("Image Sorter", image)
        cv2.waitKey(1)  # Allow OpenCV to refresh window

        # Get folder name from user
        while True:
            folder_name = input(
                f"Enter folder name for '{image_file}' (or 's' to skip): "
            ).strip()
            if folder_name.lower() == "s":
                print(f"Skipping '{image_file}'.")
                screw_ups_list.append(image_file)
                break

            folder_path = os.path.join(output_folder, folder_name)
            if os.path.isdir(folder_path):
                move_file(file_path, folder_path)
                print(f"Moved '{image_file}' to '{folder_name}'.")
                break
            else:
                print(f"Folder '{folder_name}' does not exist. Please try again.")
    print(screw_ups_list)

    cv2.destroyAllWindows()
    print("Sorting completed.")


# Run the sorter
sort_images()
