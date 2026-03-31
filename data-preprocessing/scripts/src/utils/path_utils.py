import os

def create_output_dirs(dir_output):
    # Ensure base directory exists
    os.makedirs(dir_output, exist_ok=True)

    # --- TRAIN ---
    # train/good (for images)
    os.makedirs(os.path.join(dir_output, "train", "good"), exist_ok=True)
    # New: train/bodymask (as a separate folder for train set)
    os.makedirs(os.path.join(dir_output, "train", "bodymask"), exist_ok=True)

    # --- VALID - GOOD ---
    os.makedirs(os.path.join(dir_output, "valid", "good", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "good", "label"), exist_ok=True)
    # New: valid/good/bodymask
    os.makedirs(os.path.join(dir_output, "valid", "good", "bodymask"), exist_ok=True)

    # --- VALID - UNGOOD ---
    os.makedirs(os.path.join(dir_output, "valid", "Ungood", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "Ungood", "label"), exist_ok=True)
    # New: valid/Ungood/bodymask
    os.makedirs(os.path.join(dir_output, "valid", "Ungood", "bodymask"), exist_ok=True)

    # --- TEST - GOOD ---
    os.makedirs(os.path.join(dir_output, "test", "good", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "good", "label"), exist_ok=True)
    # New: test/good/bodymask
    os.makedirs(os.path.join(dir_output, "test", "good", "bodymask"), exist_ok=True)

    # --- TEST - UNGOOD ---
    os.makedirs(os.path.join(dir_output, "test", "Ungood", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "Ungood", "label"), exist_ok=True)
    # New: test/Ungood/bodymask
    os.makedirs(os.path.join(dir_output, "test", "Ungood", "bodymask"), exist_ok=True)