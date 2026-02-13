import os
import cv2
import torch
import argparse
import numpy as np
import tempfile
import csv
import math
from PIL import Image, ImageOps
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
CHECKPOINT = "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class VideoLabeler:
    def __init__(self, original_video_dir, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV Setup
        self.csv_path = os.path.join(output_dir, "labels.csv")
        self.point_labels = {} 
        self.active_class_id = 0

        # 1. PRE-PROCESS: Use a Managed Temporary Directory
        self.temp_dir_manager = tempfile.TemporaryDirectory(prefix="sam2_labeling_")
        self.video_dir = self.temp_dir_manager.name
        
        print(f"--- Buffering frames to temp: {self.video_dir} ---")
        self.prepare_images(original_video_dir, self.video_dir)

        # 2. Get Frame List
        self.frame_names = sorted([
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        # Load existing CSV data if it exists
        self.load_csv_labels()
        
        print(f"Found {len(self.frame_names)} frames. Using device: {DEVICE}")

        # 3. Initialize SAM2
        print("Initializing SAM2 Video Predictor...")
        self.predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        
        # State variables
        self.current_idx = 0
        self.obj_id = 1
        
        # SAM State
        self.points = {}
        self.labels = {}
        
        # Polygon State
        self.poly_points = [] 

        # Mask State
        self.current_mask = None # Holds the active binary mask (0 or 1)
        self.loaded_mask = None
        self.is_dirty = False 
        self.current_frame_bgr = None
        
        # App Mode
        self.mode = "SAM" # Options: "SAM", "REFINE", "POLY", "POINT"
        
        # Refine (FloodFill) Settings
        self.tolerance = 20
        
        # Zoom & Pan State
        self.zoom_level = 1.0
        self.pan_x = 0  
        self.pan_y = 0  
        self.is_panning = False
        self.last_mouse_pos = (0, 0)
        self.mouse_x = 0
        self.mouse_y = 0
        
        self.img_h, self.img_w = 0, 0 # Set on first frame load
        
        # Window setup
        self.window_name = "SAM2 Labeler"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Trackbar
        cv2.createTrackbar("Timeline", self.window_name, 0, max(1, len(self.frame_names) - 1), self.trackbar_callback)

    def prepare_images(self, src_dir, dst_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)])
        
        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)
            try:
                img = Image.open(src_path)
                img = ImageOps.exif_transpose(img)
                img.save(dst_path, format='JPEG', quality=95)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    def load_csv_labels(self):
        if not os.path.exists(self.csv_path): return
        print(f"Loading labels from {self.csv_path}...")
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if len(row) < 4: continue
                    fname, cid, x, y = row[0], int(row[1]), int(row[2]), int(row[3])
                    if fname in self.frame_names:
                        idx = self.frame_names.index(fname)
                        if idx not in self.point_labels: self.point_labels[idx] = []
                        self.point_labels[idx] = [pt for pt in self.point_labels[idx] if pt[0] != cid]
                        self.point_labels[idx].append((cid, x, y))
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def save_csv_labels(self):
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                for idx in sorted(self.point_labels.keys()):
                    fname = self.frame_names[idx]
                    for (cid, x, y) in self.point_labels[idx]:
                        writer.writerow([fname, cid, x, y])
            print(f"[CSV] Saved to {self.csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    # --- Core Logic: Navigation & State Commitment ---
    def commit_state_to_sam(self):
        """
        Pushes the current visible mask into SAM2 memory bank.
        """
        if self.current_mask is not None and self.is_dirty:
            print(f"--> [Memory] Committing Frame {self.current_idx} to SAM2 state.")
            binary_mask = self.current_mask.astype(bool)
            _, _, _ = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.current_idx,
                obj_id=self.obj_id,
                mask=binary_mask
            )
            self.save_mask_to_disk() 

    def change_frame(self, new_idx):
        if new_idx == self.current_idx: return
        
        # 1. Commit current work to memory before leaving
        self.commit_state_to_sam()
        
        # 2. Update Index
        self.current_idx = new_idx
        cv2.setTrackbarPos("Timeline", self.window_name, self.current_idx)
        
        # 3. Reset Local State
        self.current_mask = None
        self.is_dirty = False
        self.poly_points = []
        
        # 4. Load
        self.load_frame_data()
        self.show_frame()

    def trackbar_callback(self, pos):
        if pos != self.current_idx:
            self.change_frame(pos)

    # --- Coordinate Mapping ---
    def win_to_img(self, x, y):
        scale = 1.0 / self.zoom_level
        img_x = self.pan_x + x * scale
        img_y = self.pan_y + y * scale
        return int(img_x), int(img_y)

    def img_to_win(self, x, y):
        scale = self.zoom_level
        win_x = (x - self.pan_x) * scale
        win_y = (y - self.pan_y) * scale
        return int(win_x), int(win_y)

    def apply_zoom_at_cursor(self, zoom_factor):
        new_zoom = max(1.0, min(self.zoom_level * zoom_factor, 20.0))
        anchor_x, anchor_y = self.win_to_img(self.mouse_x, self.mouse_y)
        self.zoom_level = new_zoom
        self.pan_x = anchor_x - (self.mouse_x / self.zoom_level)
        self.pan_y = anchor_y - (self.mouse_y / self.zoom_level)
        self.clamp_pan()
        self.show_frame()

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

        # 1. Panning
        if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY) and self.mode != "POLY"):
            self.is_panning = True
            self.last_mouse_pos = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                scale = 1.0 / self.zoom_level
                self.pan_x -= dx * scale
                self.pan_y -= dy * scale
                self.last_mouse_pos = (x, y)
                self.clamp_pan()
                self.show_frame()

        elif event == cv2.EVENT_MBUTTONUP or (event == cv2.EVENT_LBUTTONUP and self.is_panning):
            self.is_panning = False

        # 2. Interactive Clicks
        elif not self.is_panning:
            ix, iy = self.win_to_img(x, y)
            
            if self.mode == "SAM":
                if event == cv2.EVENT_LBUTTONDOWN: self.add_sam_point(ix, iy, 1)
                elif event == cv2.EVENT_RBUTTONDOWN: self.add_sam_point(ix, iy, 0)
            
            elif self.mode == "REFINE":
                if event == cv2.EVENT_LBUTTONDOWN: self.apply_flood_fill(ix, iy, add=True)
                elif event == cv2.EVENT_RBUTTONDOWN: self.apply_flood_fill(ix, iy, add=False)

            elif self.mode == "POLY":
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.poly_points.append((ix, iy))
                    self.show_frame()
                elif event == cv2.EVENT_RBUTTONDOWN:
                    is_subtract = (flags & cv2.EVENT_FLAG_SHIFTKEY)
                    self.apply_poly_mask(is_subtract)
            
            elif self.mode == "POINT":
                if event == cv2.EVENT_LBUTTONDOWN: self.add_class_point(ix, iy)
                elif event == cv2.EVENT_RBUTTONDOWN: self.remove_nearest_class_point(ix, iy)

    def clamp_pan(self):
        if self.img_w == 0: return
        view_w = self.img_w / self.zoom_level
        view_h = self.img_h / self.zoom_level
        self.pan_x = max(0, min(self.pan_x, max(0, self.img_w - view_w)))
        self.pan_y = max(0, min(self.pan_y, max(0, self.img_h - view_h)))

    # --- Mode 4: Polygon Logic ---
    def apply_poly_mask(self, is_subtract):
        if len(self.poly_points) < 3:
            print("[Poly] Need at least 3 points to close.")
            return

        if self.current_mask is None:
            if self.loaded_mask is not None: self.current_mask = self.loaded_mask.copy()
            else: self.current_mask = np.zeros((self.img_h, self.img_w), dtype=bool)

        poly_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(poly_mask, [pts], 1)
        poly_bool = (poly_mask > 0)
        
        if is_subtract:
            self.current_mask = np.logical_and(self.current_mask, np.logical_not(poly_bool))
            print("[Poly] Region subtracted.")
        else:
            self.current_mask = np.logical_or(self.current_mask, poly_bool)
            print("[Poly] Region added.")

        self.is_dirty = True
        self.poly_points = [] 
        self.show_frame()

    # --- Mode 3: Point Classification Logic ---
    def add_class_point(self, x, y):
        if x < 0 or y < 0 or x >= self.img_w or y >= self.img_h: return
        if self.current_idx not in self.point_labels: self.point_labels[self.current_idx] = []
        
        self.point_labels[self.current_idx] = [pt for pt in self.point_labels[self.current_idx] if pt[0] != self.active_class_id]
        self.point_labels[self.current_idx].append((self.active_class_id, x, y))
        print(f"[Point] Set Class {self.active_class_id} at ({x}, {y})")
        self.save_csv_labels()
        self.show_frame()

    def remove_nearest_class_point(self, x, y):
        if self.current_idx not in self.point_labels: return
        pts = self.point_labels[self.current_idx]
        if not pts: return
        
        min_dist = float('inf')
        min_idx = -1
        for i, (cid, px, py) in enumerate(pts):
            dist = math.sqrt((px - x)**2 + (py - y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        if min_dist < (20 / self.zoom_level):
            removed = pts.pop(min_idx)
            print(f"[Point] Removed Class {removed[0]}")
            self.save_csv_labels()
            self.show_frame()

    # --- Mode 1 & 2 Logic (SAM / Flood) ---
    def add_sam_point(self, x, y, label):
        if x < 0 or y < 0 or x >= self.img_w or y >= self.img_h: return
        if self.current_idx not in self.points:
            self.points[self.current_idx] = []
            self.labels[self.current_idx] = []
        self.points[self.current_idx].append([x, y])
        self.labels[self.current_idx].append(label)
        self.is_dirty = True
        self.update_sam_prediction()

    def update_sam_prediction(self):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_idx,
            obj_id=self.obj_id,
            points=np.array(self.points[self.current_idx], dtype=np.float32),
            labels=np.array(self.labels[self.current_idx], dtype=np.int32),
        )
        if self.obj_id in out_obj_ids:
            idx = out_obj_ids.index(self.obj_id)
            self.current_mask = (out_mask_logits[idx] > 0.0).cpu().numpy().squeeze()
            self.loaded_mask = None 
            self.show_frame()

    def apply_flood_fill(self, x, y, add=True):
        if x < 0 or y < 0 or x >= self.img_w or y >= self.img_h: return
        if self.current_frame_bgr is None: return

        if self.current_mask is None:
            if self.loaded_mask is not None: self.current_mask = self.loaded_mask.copy()
            else: self.current_mask = np.zeros((self.img_h, self.img_w), dtype=bool)

        h, w = self.current_frame_bgr.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        _, _, mask_roi, _ = cv2.floodFill(self.current_frame_bgr, flood_mask, (x, y), 0, (self.tolerance,)*3, (self.tolerance,)*3, flags)
        region_mask = mask_roi[1:-1, 1:-1] > 0

        if add: self.current_mask = np.logical_or(self.current_mask, region_mask)
        else: self.current_mask = np.logical_and(self.current_mask, np.logical_not(region_mask))
        self.is_dirty = True
        self.show_frame()

    # --- File I/O ---
    def save_mask_to_disk(self):
        mask_to_save = self.current_mask if self.current_mask is not None else self.loaded_mask
        if mask_to_save is None: return
        frame_name = self.frame_names[self.current_idx]
        mask_name = os.path.splitext(frame_name)[0] + ".png"
        save_path = os.path.join(self.output_dir, mask_name)
        mask_img = (mask_to_save * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(save_path)
        if self.is_dirty: print(f"[Save] Frame {self.current_idx} -> {mask_name}")
        self.is_dirty = False

    def track_and_propagate(self):
            print(f"\n--> Tracking initiated from Frame {self.current_idx}...")
            
            # Ensure CURRENT frame edits are in memory before propagating
            self.commit_state_to_sam()

            # Now propagate using the Memory Bank (which includes all visited & committed frames)
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.current_idx):
                if self.obj_id in out_obj_ids:
                    idx = out_obj_ids.index(self.obj_id)
                    mask = (out_mask_logits[idx] > 0.0).cpu().numpy().squeeze()
                    frame_name = self.frame_names[out_frame_idx]
                    mask_name = os.path.splitext(frame_name)[0] + ".png"
                    save_path = os.path.join(self.output_dir, mask_name)
                    Image.fromarray((mask * 255).astype(np.uint8)).save(save_path)
            print("--> Tracking complete.")
            self.load_frame_data()
            self.show_frame()

    def load_frame_data(self):
        frame_path = os.path.join(self.video_dir, self.frame_names[self.current_idx])
        self.current_frame_bgr = cv2.imread(frame_path)
        if self.img_h == 0 and self.current_frame_bgr is not None:
            self.img_h, self.img_w = self.current_frame_bgr.shape[:2]
        
        mask_name = os.path.splitext(self.frame_names[self.current_idx])[0] + ".png"
        mask_path = os.path.join(self.output_dir, mask_name)
        
        if os.path.exists(mask_path):
            loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.loaded_mask = (loaded > 127)
        else:
            self.loaded_mask = None

    def show_frame(self):
        if self.current_frame_bgr is None: return
        display_img = self.current_frame_bgr.copy()
        
        active_mask = None
        color_bgr = None
        if self.current_mask is not None:
            active_mask = self.current_mask
            color_bgr = [0, 255, 255] 
        elif self.loaded_mask is not None:
            active_mask = self.loaded_mask
            color_bgr = [255, 0, 128]

        # Apply Mask Overlay
        if active_mask is not None:
            if active_mask.shape != display_img.shape[:2]:
                active_mask = cv2.resize(active_mask.astype(np.uint8), (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST) > 0
            overlay = np.zeros_like(display_img)
            overlay[:] = color_bgr 
            blended_full = cv2.addWeighted(display_img, 0.5, overlay, 0.7, 0)
            display_img[active_mask] = blended_full[active_mask]

        # Zoom & Crop
        view_w = max(1, int(self.img_w / self.zoom_level))
        view_h = max(1, int(self.img_h / self.zoom_level))
        x1 = int(self.pan_x)
        y1 = int(self.pan_y)
        x2 = min(x1 + view_w, self.img_w)
        y2 = min(y1 + view_h, self.img_h)
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1

        cropped_view = display_img[y1:y2, x1:x2]
        final_view = cv2.resize(cropped_view, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # Draw SAM Points
        if self.mode == "SAM" and self.current_idx in self.points:
            for pt, lbl in zip(self.points[self.current_idx], self.labels[self.current_idx]):
                wx, wy = self.img_to_win(pt[0], pt[1])
                if 0 <= wx < self.img_w and 0 <= wy < self.img_h:
                    c = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                    cv2.circle(final_view, (wx, wy), 8, (255, 255, 255), -1) 
                    cv2.circle(final_view, (wx, wy), 6, c, -1)

        # Draw CLASSIFICATION Points
        if self.current_idx in self.point_labels:
            for (cid, px, py) in self.point_labels[self.current_idx]:
                wx, wy = self.img_to_win(px, py)
                if 0 <= wx < self.img_w and 0 <= wy < self.img_h:
                    color = (255, 165, 0)
                    size = 10
                    cv2.line(final_view, (wx - size, wy), (wx + size, wy), color, 2)
                    cv2.line(final_view, (wx, wy - size), (wx, wy + size), color, 2)
                    cv2.putText(final_view, str(cid), (wx+5, wy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ACTIVE POLYGON
        if self.mode == "POLY" and len(self.poly_points) > 0:
            win_pts = []
            for (px, py) in self.poly_points:
                win_pts.append(self.img_to_win(px, py))
            win_pts = np.array(win_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(final_view, [win_pts], False, (0, 255, 0), 2)
            for wp in win_pts:
                cv2.circle(final_view, (wp[0][0], wp[0][1]), 4, (0, 255, 0), -1)

        # UI Text
        status = f"F:{self.current_idx} | Mode: {self.mode}"
        if self.mode == "REFINE": status += f" (Tol:{self.tolerance})"
        elif self.mode == "POINT": status += f" (Class:{self.active_class_id})"
        elif self.mode == "POLY": status += f" (Pts:{len(self.poly_points)})"
        
        if self.is_dirty: status += " [UNSAVED]"
        else: status += " [SAVED]"
        
        cv2.putText(final_view, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        instr = "TAB: Mode | "
        if self.mode == "SAM": instr += "L/R: Pts"
        elif self.mode == "REFINE": instr += "L/R: Fill"
        elif self.mode == "POINT": instr += "L:Add R:Del"
        elif self.mode == "POLY": instr += "L:Pt R:Fill Bksp:Undo"
        
        cv2.putText(final_view, instr, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(final_view, "Nav: Arrows | Esc: Cancel Poly", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(self.window_name, final_view)

    def run(self):
        print("\n=== CONTROLS ===")
        print(" TAB        : Cycle (SAM -> REFINE -> POLY -> POINT)")
        print(" 'z' / 'x'  : Zoom In / Out")
        print(" Mid-Drag   : Pan")
        print(" Arrows     : Navigation (Auto-Saves to Memory)")
        print(" 't'        : Track (SAM)")
        print(" 's'        : Save Mask")
        print(" -- Poly Mode --")
        print(" L-Click    : Add Vertex")
        print(" Bksp/Del   : Undo Last Vertex")
        print(" R-Click    : Close & Fill")
        print(" Sh+R-Click : Close & Erase")
        print(" Esc        : Cancel Poly")
        print("================\n")

        self.load_frame_data()
        self.show_frame()

        while True:
            raw_key = cv2.waitKey(20)
            if raw_key == -1: continue 
            key_char = raw_key & 0xFF

            if key_char == ord('q') or raw_key == 27: # q or ESC
                if self.mode == "POLY" and len(self.poly_points) > 0:
                    self.poly_points = []
                    self.show_frame()
                elif raw_key == 27:
                    break
                elif key_char == ord('q'):
                    break
            
            # --- UNDO POLY POINT (Backspace / Del) ---
            elif key_char in [8, 127]:
                if self.mode == "POLY" and len(self.poly_points) > 0:
                    self.poly_points.pop()
                    self.show_frame()

            elif key_char == 9: # Tab
                if self.mode == "SAM": self.mode = "REFINE"
                elif self.mode == "REFINE": self.mode = "POLY"
                elif self.mode == "POLY": 
                    self.poly_points = [] 
                    self.mode = "POINT"
                else: self.mode = "SAM"
                self.show_frame()

            # Nav - USES NEW CHANGE_FRAME LOGIC
            elif key_char in [ord('a'), ord('p'), ord(',')] or raw_key in [81, 2, 65361, 2424832]:
                self.change_frame(max(self.current_idx - 1, 0))
            elif key_char in [ord('d'), ord('n'), ord('.')] or raw_key in [83, 3, 65363, 2555904]: 
                self.change_frame(min(self.current_idx + 1, len(self.frame_names) - 1))

            # Class (0-9)
            elif 48 <= key_char <= 57:
                self.active_class_id = key_char - 48
                if self.mode == "POINT": self.show_frame()

            # Zoom / Tools
            elif key_char == ord('z'): self.apply_zoom_at_cursor(1.25)
            elif key_char == ord('x'): self.apply_zoom_at_cursor(0.8)
            elif key_char == ord(']'): 
                self.tolerance = min(self.tolerance + 5, 255)
                self.show_frame()
            elif key_char == ord('['): 
                self.tolerance = max(self.tolerance - 5, 1)
                self.show_frame()
            elif key_char == ord('t'): self.track_and_propagate()
            elif key_char == ord('s'): self.save_mask_to_disk()
            elif key_char == ord('r'):
                if self.current_idx in self.points:
                    del self.points[self.current_idx]
                    del self.labels[self.current_idx]
                self.predictor.reset_state(self.inference_state)
                self.current_mask = None
                self.is_dirty = False
                self.load_frame_data()
                self.show_frame()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    labeler = VideoLabeler(args.input_dir, args.output_dir)
    labeler.run()