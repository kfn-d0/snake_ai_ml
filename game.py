import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREY = (40, 40, 40)
GRAY_ZONE = (128, 128, 128)
LIGHT_GRAY = (160, 160, 160)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480, phase=1, title="Snake AI"):
        self.w = w
        self.h = h
        self.display_w = w + 300
        self.display_h = h
        self.phase = phase
        # display
        self.display = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.gray_zone = []
        self.obstacles = []
        self.reset()

    def _generate_map_layout(self):
        self.gray_zone = []
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE

        # layout style
        layout_style = random.choice(['octagon', 'cross', 'thick_border', 'classic'])

        if layout_style == 'octagon':
            corner_cut = random.randint(3, 7)
            for x in range(cols):
                for y in range(rows):
                    if (x + y < corner_cut) or \
                       ((cols - 1 - x) + y < corner_cut) or \
                       (x + (rows - 1 - y) < corner_cut) or \
                       ((cols - 1 - x) + (rows - 1 - y) < corner_cut):
                        self.gray_zone.append(Point(x * BLOCK_SIZE, y * BLOCK_SIZE))
                        
        elif layout_style == 'cross':
            cut_depth = random.randint(2, 6)
            cut_width = random.randint(8, 14)
            mid_x, mid_y = cols // 2, rows // 2
            
            for x in range(cols):
                for y in range(rows):
                    if (abs(x - mid_x) > cut_width // 2) and (y < cut_depth or y >= rows - cut_depth):
                        self.gray_zone.append(Point(x * BLOCK_SIZE, y * BLOCK_SIZE))
                    elif (abs(y - mid_y) > cut_width // 2) and (x < cut_depth or x >= cols - cut_depth):
                        self.gray_zone.append(Point(x * BLOCK_SIZE, y * BLOCK_SIZE))

        elif layout_style == 'thick_border':
            b_thickness = random.randint(2, 4)
            for x in range(cols):
                for y in range(rows):
                    if x < b_thickness or x >= cols - b_thickness or y < b_thickness or y >= rows - b_thickness:
                        self.gray_zone.append(Point(x * BLOCK_SIZE, y * BLOCK_SIZE))

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
        self.seed_val = seed
        random.seed(seed)
        
        self.score = 0
        self.frame_iteration = 0
        
        self._generate_map_layout()

        # obstacles
        self.obstacles = []
        self._place_obstacles()

        # Spawn dinamico
        self.direction = Direction.RIGHT
        while True:
            hx = random.randint(2, (self.w - BLOCK_SIZE)//BLOCK_SIZE - 2) * BLOCK_SIZE
            hy = random.randint(1, (self.h - BLOCK_SIZE)//BLOCK_SIZE - 1) * BLOCK_SIZE
            
            pts = [Point(hx, hy), Point(hx - BLOCK_SIZE, hy), Point(hx - 2 * BLOCK_SIZE, hy)]
            valid = True
            for pt in pts:
                if pt in self.gray_zone or pt in self.obstacles:
                    valid = False
            if valid:
                self.head = pts[0]
                self.snake = pts
                break

        self.food = None
        self._place_food()

    def _place_obstacles(self):
        num_obstacles = 4 + random.randint(0, 5) + (self.score // 5)
        for _ in range(num_obstacles):
            for _ in range(100):
                obs = Point(random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE,
                            random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE)
                
                if obs not in self.gray_zone:
                    self.obstacles.append(obs)
                    break

    def _place_food(self):
        for _ in range(100):
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles and self.food not in self.gray_zone:
                break

    def play_step(self, action, debug_info=None):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        dist_before = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            
            dist_after = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
            
            if dist_after < dist_before:
                reward = 0.5
            else:
                reward = -0.1

        self._update_ui(debug_info)
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.gray_zone:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True

        return False

    def _update_ui(self, debug_info=None):
        self.display.fill(BLACK)
        
        pygame.draw.rect(self.display, (20, 20, 30), pygame.Rect(self.w, 0, self.display_w - self.w, self.display_h))
        pygame.draw.line(self.display, WHITE, (self.w, 0), (self.w, self.display_h), 2)

        for gz in self.gray_zone:
            pygame.draw.rect(self.display, GRAY_ZONE, pygame.Rect(gz.x, gz.y, BLOCK_SIZE, BLOCK_SIZE))

        for obs in self.obstacles:
            pygame.draw.rect(self.display, LIGHT_GRAY, pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [self.w + 10, 10])
        
        if debug_info:
            small_font = pygame.font.SysFont('arial', 15)
            y_offset = 50
            for key, val in debug_info.items():
                debug_text = small_font.render(f"{key}: {val}", True, (255, 255, 100))
                self.display.blit(debug_text, [self.w + 10, y_offset])
                y_offset += 25

        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
