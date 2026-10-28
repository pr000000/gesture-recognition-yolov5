# Author: Peiran (Cynthia) Jing | GitHub: https://github.com/pr000000
# Image processing tool (Tkinter + OpenCV)
import cv2                                   # OpenCV image processing
import numpy as np                           # numerical computing
import os                                    # file paths
from tkinter import *                        # Tkinter GUI
from tkinter import filedialog, messagebox   # file dialogs, message boxes
from PIL import Image, ImageTk               # convert images for display in Tk

# Main window
root = Tk()
root.title("Image Processing Tool")
root.geometry("1200x700")

# Image state (original and processed)
original_image = None
processed_image = None

# Image display area
img_label = Label(root)
img_label.pack()


def display_image(img):
    """Show an OpenCV image in the Tkinter window."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR; PIL expects RGB
    img_pil = Image.fromarray(img_rgb)              # NumPy array -> PIL image
    img_pil = img_pil.resize((600, 400))            # fit the display area
    img_tk = ImageTk.PhotoImage(img_pil)            # PIL -> Tk-compatible image
    img_label.configure(image=img_tk)
    img_label.image = img_tk                        # keep a reference so it isn't garbage-collected


def load_image():
    """Open an image file and display it."""
    global original_image, processed_image
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            file_path = os.path.abspath(file_path)
            # np.fromfile + imdecode instead of cv2.imread: supports non-ASCII
            # (e.g. Chinese) characters in Windows file paths
            img_array = np.fromfile(file_path, dtype=np.uint8)
            original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if original_image is None:
                messagebox.showwarning("Error", "Could not read the image. Please try another file.")
                return
            processed_image = original_image.copy()
            display_image(processed_image)
            apply_changes()  # apply the current slider settings
        except Exception as e:
            messagebox.showerror("Load failed", str(e))


def apply_changes():
    """Apply all slider settings to the original image (non-destructive)."""
    global processed_image
    if original_image is None:
        return

    # Read slider values
    brightness = brightness_scale.get()      # 0-100 (default 50)
    contrast = contrast_scale.get()          # 0-100 (default 50)
    hue = hue_scale.get()                    # -90 to 90
    saturation = saturation_scale.get()      # 0-100 (default 50)
    scale = scale_scale.get()                # 10%-200%
    angle = rotate_scale.get()               # -180° to 180°
    denoise = denoise_scale.get()            # 0-20
    sharpen = sharpen_scale.get()            # 0-5

    # Always start from the original, in float32 to avoid overflow/precision loss
    img = original_image.copy().astype(np.float32)

    # -------- Brightness & contrast: linear transform --------
    img = img * (contrast / 50) + (brightness - 50)
    img = np.clip(img, 0, 255)  # keep values in [0, 255]

    # -------- Hue & saturation in HSV space --------
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue) % 180  # shift hue (OpenCV H range is 0-179)
    img_hsv[:, :, 1] *= (saturation / 50)              # scale saturation
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # -------- Scaling --------
    scale_factor = scale / 100.0
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # -------- Rotation about the centre --------
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # -------- Denoising (Gaussian blur) --------
    if denoise > 0:
        k = int(denoise // 2) * 2 + 1  # kernel size must be odd
        img = cv2.GaussianBlur(img, (k, k), 0)

    # -------- Sharpening (Laplacian-style kernel) --------
    if sharpen > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpen, -1],
                           [0, -1, 0]])  # larger centre weight = stronger sharpening
        img = cv2.filter2D(img, -1, kernel)

    # Store and display the result
    processed_image = np.clip(img, 0, 255).astype(np.uint8)
    display_image(processed_image)


def save_image():
    """Save the processed image to a file."""
    if processed_image is None:
        messagebox.showwarning("Notice", "There is no processed image to save yet.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            ext = os.path.splitext(file_path)[1]
            # imencode + tofile instead of cv2.imwrite: supports non-ASCII paths
            result, encoded_img = cv2.imencode(ext, processed_image)
            if result:
                encoded_img.tofile(file_path)
                messagebox.showinfo("Saved", f"Image saved to:\n{file_path}")
            else:
                raise ValueError("Image encoding failed")
        except Exception as e:
            messagebox.showerror("Save failed", f"Failed to save the image:\n{e}")


def create_slider(label_text, row, from_, to, initial=50):
    """Create a labelled slider that re-applies all changes when moved."""
    Label(control_frame, text=label_text).grid(row=row, column=0)
    scale = Scale(control_frame, from_=from_, to=to, orient=HORIZONTAL,
                  command=lambda x: apply_changes())
    scale.set(initial)
    scale.grid(row=row, column=1)
    return scale


# -------- Control panel --------
control_frame = Frame(root)
control_frame.pack()

# Load / save buttons
Button(control_frame, text="Load Image", command=load_image).grid(row=0, column=0, pady=10)
Button(control_frame, text="Save Image", command=save_image).grid(row=0, column=1, padx=10)

# -------- Sliders --------
brightness_scale = create_slider("Brightness", 1, 0, 100, 50)
contrast_scale = create_slider("Contrast", 2, 0, 100, 50)
hue_scale = create_slider("Hue", 3, -90, 90, 0)
saturation_scale = create_slider("Saturation", 4, 0, 100, 50)
scale_scale = create_slider("Scale (%)", 5, 10, 200, 100)
rotate_scale = create_slider("Rotation (°)", 6, -180, 180, 0)
denoise_scale = create_slider("Denoise", 7, 0, 20, 0)
sharpen_scale = create_slider("Sharpen", 8, 0, 5, 0)

# Start the Tkinter event loop
root.mainloop()
