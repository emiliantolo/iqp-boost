import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset

class BinaryShapesDataset(BinaryDataset):
    """
    Generates a dataset of 2D binary grids containing geometric shapes.
    Supported shapes: squares, rectangles, bars (lines), triangles, crosses, diamonds, L-shapes.
    Each sample contains exactly one shape, and the shapes are "hollow" (outlines only).
    """

    def __init__(self, grid_shape: tuple[int, int] = (5, 5), shape_types: list[str] = None):
        """
        Initialize the Binary Shapes dataset generator.
        
        Args:
            grid_shape (tuple[int, int]): A tuple (rows, cols) defining the 2D lattice. Default is (5, 5).
            shape_types (list[str], optional): Allowed shapes. Defaults to a comprehensive list.
        """
        super().__init__()
        self.grid_shape = grid_shape
        # n_shapes is removed to strictly enforce 1 shape per grid
        self.shape_types = shape_types if shape_types is not None else [
            'square', 'rectangle', 'bar_horizontal', 'bar_vertical', 
            'triangle', 'cross', 'diamond', 'l_shape', 'u_shape', 'x_shape'
        ]
        self.N = grid_shape[0] * grid_shape[1]
        
        # Pre-generate and cache all unique valid shapes for membership testing
        self._valid_patterns = None  # Lazy-loaded

    def _draw_square_exact(self, grid: np.ndarray, r: int, c: int, side: int):
        grid[r, c:c+side] = 1 # Top
        grid[r+side-1, c:c+side] = 1 # Bottom
        grid[r:r+side, c] = 1 # Left
        grid[r:r+side, c+side-1] = 1 # Right

    def _draw_square(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        side = rng.integers(3, min(rows, cols) + 1) # Min 3x3 to actually have a hollow center
        r = rng.integers(0, rows - side + 1)
        c = rng.integers(0, cols - side + 1)
        self._draw_square_exact(grid, r, c, side)
        
    def _draw_rectangle_exact(self, grid: np.ndarray, r: int, c: int, h: int, w: int):
        grid[r, c:c+w] = 1 # Top
        grid[r+h-1, c:c+w] = 1 # Bottom
        grid[r:r+h, c] = 1 # Left
        grid[r:r+h, c+w-1] = 1 # Right

    def _draw_rectangle(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        h = rng.integers(3, rows + 1) # Min 3 to be hollow
        w = rng.integers(3, cols + 1) # Min 3 to be hollow
        r = rng.integers(0, rows - h + 1)
        c = rng.integers(0, cols - w + 1)
        self._draw_rectangle_exact(grid, r, c, h, w)
        
    def _draw_bar_horizontal_exact(self, grid: np.ndarray, r: int, c: int, length: int):
        grid[r, c:c+length] = 1

    def _draw_bar_horizontal(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        length = rng.integers(2, cols + 1)
        r = rng.integers(0, rows)
        c = rng.integers(0, cols - length + 1)
        self._draw_bar_horizontal_exact(grid, r, c, length)
            
    def _draw_bar_vertical_exact(self, grid: np.ndarray, r: int, c: int, length: int):
        grid[r:r+length, c] = 1

    def _draw_bar_vertical(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        length = rng.integers(2, rows + 1)
        r = rng.integers(0, rows - length + 1)
        c = rng.integers(0, cols)
        self._draw_bar_vertical_exact(grid, r, c, length)

    def _draw_triangle_exact(self, grid: np.ndarray, r: int, c: int, base: int, direction: str):
        if direction == 'bl':
            grid[r:r+base, c] = 1 # Left leg
            grid[r+base-1, c:c+base] = 1 # Bottom leg
            for i in range(base): grid[r+i, c+base-1-i] = 1 # Hypotenuse
        elif direction == 'br':
            grid[r:r+base, c+base-1] = 1 # Right leg
            grid[r+base-1, c:c+base] = 1 # Bottom leg
            for i in range(base): grid[r+i, c+i] = 1
        elif direction == 'tl':
            grid[r:r+base, c] = 1 # Left leg
            grid[r, c:c+base] = 1 # Top leg
            for i in range(base): grid[r+i, c+i] = 1
        elif direction == 'tr':
            grid[r:r+base, c+base-1] = 1 # Right leg
            grid[r, c:c+base] = 1 # Top leg
            for i in range(base): grid[r+i, c+base-1-i] = 1

    def _draw_triangle(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        # Right triangle outline
        base = rng.integers(3, min(rows, cols) + 1)
        r = rng.integers(0, rows - base + 1)
        c = rng.integers(0, cols - base + 1)
        direction = rng.choice(['bl', 'br', 'tl', 'tr'])
        self._draw_triangle_exact(grid, r, c, base, direction)

    def _draw_cross_exact(self, grid: np.ndarray, r_center: int, c_center: int, size: int):
        grid[r_center, c_center - size // 2 : c_center + size // 2 + 1] = 1
        grid[r_center - size // 2 : r_center + size // 2 + 1, c_center] = 1

    def _draw_cross(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        size = rng.integers(3, min(rows, cols) + 1)
        if size % 2 == 0:
            size -= 1 # Keep odd for perfect center intersection
        r_center = rng.integers(size // 2, rows - size // 2)
        c_center = rng.integers(size // 2, cols - size // 2)
        self._draw_cross_exact(grid, r_center, c_center, size)

    def _draw_x_shape_exact(self, grid: np.ndarray, r_center: int, c_center: int, size: int):
        for i in range(-size//2, size//2 + 1):
            try:
                grid[r_center + i, c_center + i] = 1
                grid[r_center + i, c_center - i] = 1
            except IndexError:
                pass

    def _draw_x_shape(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        size = rng.integers(3, min(rows, cols) + 1)
        if size % 2 == 0:
            size -= 1
        r_center = rng.integers(size // 2, rows - size // 2)
        c_center = rng.integers(size // 2, cols - size // 2)
        self._draw_x_shape_exact(grid, r_center, c_center, size)

    def _draw_diamond_exact(self, grid: np.ndarray, r_center: int, c_center: int, size: int):
        radius = size // 2
        for i in range(radius + 1):
            try:
                grid[r_center - radius + i, c_center + i] = 1
                grid[r_center + radius - i, c_center + i] = 1
                grid[r_center - radius + i, c_center - i] = 1
                grid[r_center + radius - i, c_center - i] = 1
            except IndexError:
                pass

    def _draw_diamond(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        size = rng.integers(3, min(rows, cols) + 1)
        if size % 2 == 0:
            size -= 1
        r_center = rng.integers(size // 2, rows - size // 2)
        c_center = rng.integers(size // 2, cols - size // 2)
        self._draw_diamond_exact(grid, r_center, c_center, size)

    def _draw_l_shape_exact(self, grid: np.ndarray, r: int, c: int, h: int, w: int, direction: str):
        if direction == 'bl':
            grid[r:r+h, c] = 1 # Vertical left
            grid[r+h-1, c:c+w] = 1 # Horizontal bottom
        elif direction == 'br':
            grid[r:r+h, c+w-1] = 1 # Vertical right
            grid[r+h-1, c:c+w] = 1 # Horizontal bottom
        elif direction == 'tl':
            grid[r:r+h, c] = 1 # Vertical left
            grid[r, c:c+w] = 1 # Horizontal top
        elif direction == 'tr':
            grid[r:r+h, c+w-1] = 1 # Vertical right
            grid[r, c:c+w] = 1 # Horizontal top

    def _draw_l_shape(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        h = rng.integers(3, rows + 1)
        w = rng.integers(3, cols + 1)
        r = rng.integers(0, rows - h + 1)
        c = rng.integers(0, cols - w + 1)
        direction = rng.choice(['bl', 'br', 'tl', 'tr'])
        self._draw_l_shape_exact(grid, r, c, h, w, direction)

    def _draw_u_shape_exact(self, grid: np.ndarray, r: int, c: int, h: int, w: int, direction: str):
        if direction == 'up': # Open top
            grid[r:r+h, c] = 1 # Left 
            grid[r:r+h, c+w-1] = 1 # Right
            grid[r+h-1, c:c+w] = 1 # Bottom
        elif direction == 'down': # Open bottom
            grid[r:r+h, c] = 1
            grid[r:r+h, c+w-1] = 1
            grid[r, c:c+w] = 1 # Top
        elif direction == 'right': # Open right
            grid[r, c:c+w] = 1
            grid[r+h-1, c:c+w] = 1
            grid[r:r+h, c] = 1 # Left
        elif direction == 'left': # Open left
            grid[r, c:c+w] = 1
            grid[r+h-1, c:c+w] = 1
            grid[r:r+h, c+w-1] = 1 # Right

    def _draw_u_shape(self, grid: np.ndarray, rng: np.random.Generator):
        rows, cols = self.grid_shape
        h = rng.integers(3, rows + 1)
        w = rng.integers(3, cols + 1)
        r = rng.integers(0, rows - h + 1)
        c = rng.integers(0, cols - w + 1)
        direction = rng.choice(['up', 'down', 'left', 'right'])
        self._draw_u_shape_exact(grid, r, c, h, w, direction)

    def _generate_all(self) -> np.ndarray:
        rows, cols = self.grid_shape
        all_grids = []

        def add_grid(draw_fn, *args):
            grid = np.zeros(self.grid_shape, dtype=np.int8)
            draw_fn(grid, *args)
            if np.any(grid):
                all_grids.append(grid.flatten())

        for shape_type in self.shape_types:
            if shape_type == 'square':
                max_side = min(rows, cols)
                if max_side >= 3:
                    for side in range(3, max_side + 1):
                        for r in range(rows - side + 1):
                            for c in range(cols - side + 1):
                                add_grid(self._draw_square_exact, r, c, side)

            elif shape_type == 'rectangle':
                for h in range(3, rows + 1):
                    for w in range(3, cols + 1):
                        for r in range(rows - h + 1):
                            for c in range(cols - w + 1):
                                add_grid(self._draw_rectangle_exact, r, c, h, w)

            elif shape_type == 'bar_horizontal':
                for length in range(2, cols + 1):
                    for r in range(rows):
                        for c in range(cols - length + 1):
                            add_grid(self._draw_bar_horizontal_exact, r, c, length)

            elif shape_type == 'bar_vertical':
                for length in range(2, rows + 1):
                    for r in range(rows - length + 1):
                        for c in range(cols):
                            add_grid(self._draw_bar_vertical_exact, r, c, length)

            elif shape_type == 'triangle':
                max_base = min(rows, cols)
                if max_base >= 3:
                    for base in range(3, max_base + 1):
                        for r in range(rows - base + 1):
                            for c in range(cols - base + 1):
                                for direction in ['bl', 'br', 'tl', 'tr']:
                                    add_grid(self._draw_triangle_exact, r, c, base, direction)

            elif shape_type in ['cross', 'diamond', 'x_shape']:
                max_size = min(rows, cols)
                if max_size >= 3:
                    for size in range(3, max_size + 1):
                        if size % 2 == 0:
                            continue
                        for r_center in range(size // 2, rows - size // 2):
                            for c_center in range(size // 2, cols - size // 2):
                                if shape_type == 'cross':
                                    add_grid(self._draw_cross_exact, r_center, c_center, size)
                                elif shape_type == 'diamond':
                                    add_grid(self._draw_diamond_exact, r_center, c_center, size)
                                elif shape_type == 'x_shape':
                                    add_grid(self._draw_x_shape_exact, r_center, c_center, size)

            elif shape_type == 'l_shape':
                for h in range(3, rows + 1):
                    for w in range(3, cols + 1):
                        for r in range(rows - h + 1):
                            for c in range(cols - w + 1):
                                for direction in ['bl', 'br', 'tl', 'tr']:
                                    add_grid(self._draw_l_shape_exact, r, c, h, w, direction)

            elif shape_type == 'u_shape':
                for h in range(3, rows + 1):
                    for w in range(3, cols + 1):
                        for r in range(rows - h + 1):
                            for c in range(cols - w + 1):
                                for direction in ['up', 'down', 'left', 'right']:
                                    add_grid(self._draw_u_shape_exact, r, c, h, w, direction)

        if not all_grids:
            return np.zeros((0, self.N), dtype=np.int8)

        # Remove duplicates (e.g. from square and rectangle overlaps, or identical configurations)
        all_grids_np = np.stack(all_grids)
        unique_grids = np.unique(all_grids_np, axis=0)
        return unique_grids

    def generate(self, n_samples: int = None, seed: int = None) -> np.ndarray:
        """
        Generates binary strings representing exactly one hollow shape on a 2D grid.
        If n_samples is None, generates all possible valid shapes exactly once.

        Args:
            n_samples (int, optional): Number of samples to generate. None for all possible.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Matrix of shape (num_samples, rows * cols) with elements 0 or 1.
        """
        if n_samples is None:
            self.data = self._generate_all()
            return self.data

        rng = np.random.default_rng(seed)
        data = np.zeros((n_samples, self.N), dtype=np.int8)
        
        for i in range(n_samples):
            grid = np.zeros(self.grid_shape, dtype=np.int8)
            shape_type = rng.choice(self.shape_types)
            
            if shape_type == 'square':
                self._draw_square(grid, rng)
            elif shape_type == 'rectangle':
                self._draw_rectangle(grid, rng)
            elif shape_type == 'bar_horizontal':
                self._draw_bar_horizontal(grid, rng)
            elif shape_type == 'bar_vertical':
                self._draw_bar_vertical(grid, rng)
            elif shape_type == 'triangle':
                self._draw_triangle(grid, rng)
            elif shape_type == 'cross':
                self._draw_cross(grid, rng)
            elif shape_type == 'diamond':
                self._draw_diamond(grid, rng)
            elif shape_type == 'l_shape':
                self._draw_l_shape(grid, rng)
            elif shape_type == 'u_shape':
                self._draw_u_shape(grid, rng)
            elif shape_type == 'x_shape':
                self._draw_x_shape(grid, rng)
                
            data[i] = grid.flatten()
            
        self.data = data
        return self.data

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualizes a single shapes sample as a 2D image.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.grid_shape[1] * 0.5, self.grid_shape[0] * 0.5))
            
        sample_2d = sample.reshape(self.grid_shape)
        ax.imshow(sample_2d, cmap='Blues', aspect='equal')
        ax.set_title("Binary Shapes Sample")
        
        # Add a grid to distinguish qubits
        ax.set_xticks(np.arange(-0.5, self.grid_shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()

    def _ensure_valid_patterns(self):
        """
        Lazily compute and cache the set of all unique valid hollow shapes.
        This is done once on first validity check to avoid expensive computation.
        """
        if self._valid_patterns is not None:
            return
        
        # Generate all possible unique hollow shapes
        all_shapes = self._generate_all()
        self._valid_patterns = set(map(tuple, all_shapes))
