import numpy as np
import pygame
import json
import os

def load_logs(log_file):
    """Load the agent's thought process logs from the JSON file."""
    with open(log_file, 'r') as file:
        logs = [json.loads(line) for line in file]
    return logs

def draw_grid(screen, field, cell_size):
    """Draw the game field grid."""
    rows, cols = field.shape
    for x in range(rows):
        for y in range(cols):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            color = (200, 200, 200) if field[x, y] == 0 else (100, 100, 100)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

def draw_features(screen, log, cell_size):
    """Overlay the agent's thought process features on the grid."""
    danger_map = log['danger_map']
    reachable_tiles = log['reachable_tiles']
    traps = log['traps']
    chokepoints = log['chokepoints']
    coins = log['coins']
    opponents = log['opponents']
    position = log['position']

    # Draw danger zones
    for t, danger_tiles in enumerate(danger_map):
        for x, y in danger_tiles:
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255, 0, 0), rect, 2)

    # Draw reachable tiles
    for x, y in reachable_tiles:
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 255, 0), rect, 2)

    # Draw traps
    for x, y in traps:
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (255, 255, 0), rect, 2)

    # Draw chokepoints
    for x, y in chokepoints:
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 0, 255), rect, 2)

    # Draw coins
    for x, y in coins:
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.circle(screen, (255, 215, 0), rect.center, cell_size // 4)

    # Draw opponents
    for x, y in opponents:
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.circle(screen, (128, 0, 128), rect.center, cell_size // 4)

    # Draw agent's position
    x, y = position
    rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
    pygame.draw.circle(screen, (0, 255, 255), rect.center, cell_size // 4)

def replay_viewer(log_file, field_shape):
    """Visualize the agent's thought process using pygame."""
    pygame.init()
    cell_size = 30
    screen = pygame.display.set_mode((field_shape[1] * cell_size, field_shape[0] * cell_size))
    pygame.display.set_caption("Agent Thought Process Viewer")

    logs = load_logs(log_file)
    clock = pygame.time.Clock()
    running = True
    frame = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        if frame < len(logs):
            log = logs[frame]
            field = np.array(log.get('field', [[0] * field_shape[1] for _ in range(field_shape[0])]))
            draw_grid(screen, field, cell_size)
            draw_features(screen, log, cell_size)
            frame += 1

        pygame.display.flip()
        clock.tick(10)  # Adjust speed as needed

    pygame.quit()

if __name__ == "__main__":
    log_file_path = "agent_code/simple_agent/simple_agent_thoughts.json"
    if os.path.exists(log_file_path):
        replay_viewer(log_file_path, field_shape=(17, 17))  # Adjust field shape as needed
    else:
        print("Log file not found.")
