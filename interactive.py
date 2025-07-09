import torch
from torch import nn
import pygame
import numpy as np
from config import *
from dataset import *
from model import *

# --- Configuration ---
# Use the cpu to make the program guranteed to run on all devices since interactive.py doesn't need to run extremely fast
device ="cpu"
print(f"Using device: {device}")

# Pygame Window Configuration
CANVAS_SIZE = 280  # The drawing area is 280x280 pixels
GRID_DIM = 28      # The underlying model grid is 28x28
PIXEL_SIZE = CANVAS_SIZE // GRID_DIM # Size of each grid cell on screen
UI_WIDTH = 200     # Width of the sidebar for displaying probabilities
WINDOW_WIDTH = CANVAS_SIZE + UI_WIDTH
WINDOW_HEIGHT = CANVAS_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
GREEN = (0, 200, 0)

# Load the model
print("Loading model...")
model = MNISTModel()
model.load_state_dict(torch.load('digit_model.pth', map_location=device))
model.eval() # Set the model to evaluation mode

# --- Pygame and Application Setup ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Interactive Digit Recognizer')
font = pygame.font.SysFont("sans-serif", 22)
ui_font = pygame.font.SysFont("sans-serif", 18)
drawing_tensor = torch.zeros((GRID_DIM, GRID_DIM), dtype=torch.float32, device=device)

# --- Main Application Loop ---
running = True
is_drawing = False
is_erasing = False
last_pos = None

# --- MODIFIED: New brush logic ---
def update_canvas(pos, mode):
    """Update the drawing tensor based on mouse position and mode."""
    gx = pos[0] // PIXEL_SIZE
    gy = pos[1] // PIXEL_SIZE

    if mode == 'draw':
        # Define the soft brush kernel: center pixel is brightest, neighbors are dimmer.
        # This creates a softer, anti-aliased line.
        kernel = [
            (0, 0, 1.0),   # Center pixel (full intensity)
            (0, 1, 0.4),   # Adjacent pixels (lower intensity)
            (0, -1, 0.4),
            (1, 0, 0.4),
            (-1, 0, 0.4)
        ]
        for dx, dy, intensity in kernel:
            px, py = gx + dx, gy + dy
            if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                # Use max to avoid overwriting a brighter pixel with a dimmer one
                current_val = drawing_tensor[py, px].item()
                drawing_tensor[py, px] = max(current_val, intensity)

    else: # erase
        # Use a 3x3 hard eraser for better usability
        erase_radius = 1
        for x_offset in range(-erase_radius, erase_radius + 1):
            for y_offset in range(-erase_radius, erase_radius + 1):
                px, py = gx + x_offset, gy + y_offset
                if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                    drawing_tensor[py, px] = 0.0

def draw_line(start, end, mode):
    """Draw a line on the tensor between two points using the brush."""
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]
    
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        update_canvas((x1, y1), mode)
        if x1 == x2 and y1 == y2: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x1 += sx
        if e2 < dx: err += dx; y1 += sy

# --- Instructions ---
print("\nApplication running. Draw on the black canvas.")
print("  - Left Mouse: Draw")
print("  - Right Mouse: Erase")
print("  - 'c' key: Clear canvas")

while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                drawing_tensor.fill_(0)
                print("Canvas cleared.")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: is_drawing = True
            elif event.button == 3: is_erasing = True
            last_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            is_drawing = False
            is_erasing = False
            last_pos = None
        
        elif event.type == pygame.MOUSEMOTION:
            if (is_drawing or is_erasing) and event.pos[0] < CANVAS_SIZE:
                mode = 'draw' if is_drawing else 'erase'
                if last_pos:
                    draw_line(last_pos, event.pos, mode)
                last_pos = event.pos

    # --- Prediction ---
    with torch.no_grad():
        normalized_tensor = (drawing_tensor - mnist_mean[0]) / mnist_std[0]
        output = model(normalized_tensor)
        softmax_output = torch.softmax(output, dim=1).squeeze()
        probabilities = softmax_output.cpu().numpy()
        predicted_label = np.argmax(probabilities)

    # --- Drawing and Rendering ---
    screen.fill(GREY)

    # Draw the canvas
    for i in range(GRID_DIM):
        for j in range(GRID_DIM):
            color_val = int(drawing_tensor[i, j].item() * 255)
            color = (color_val, color_val, color_val)
            pygame.draw.rect(screen, color, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
    
    # Draw the UI panel
    ui_x_start = CANVAS_SIZE
    pygame.draw.rect(screen, BLACK, (ui_x_start, 0, UI_WIDTH, WINDOW_HEIGHT))
    
    title_text = font.render("Predictions", True, WHITE)
    screen.blit(title_text, (ui_x_start + 45, 10))
    
    # Draw probability bars
    for i in range(NUM_CLASSES):
        # --- FIXED: Reduced vertical spacing to fit all 10 bars ---
        y_pos = 50 + i * 22
        prob = probabilities[i]
        bar_color = GREEN if i == predicted_label else GREY
        bar_width = int(prob * (UI_WIDTH - 60))
        
        label_text = ui_font.render(f"{i}:", True, WHITE)
        screen.blit(label_text, (ui_x_start + 10, y_pos))

        pygame.draw.rect(screen, bar_color, (ui_x_start + 35, y_pos, bar_width, 18)) # Made bars slightly thinner
        
        if prob > 0.001:
            prob_text = ui_font.render(f"{prob:.1%}", True, WHITE)
            screen.blit(prob_text, (ui_x_start + 40 + bar_width, y_pos))

    pygame.display.flip()

pygame.quit()