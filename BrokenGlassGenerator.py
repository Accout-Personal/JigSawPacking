import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
from scipy.spatial import Voronoi
import os,json
from pathlib import Path
import sys, subprocess
import shutil
import copy

DEBUG = False
try:
    import cplex
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False

CPLEX_AVAILABLE = False

def exe_dir() -> Path:
        if getattr(sys, "frozen", False):      # PyInstaller / frozen app
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent  # normal python run

BASE_DIR = exe_dir()
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

class GridJigsawGenerator:
    def __init__(self, width, height, step=1.0, min_distance=None, target_pieces=30, min_fusion=2, retry_until_success=False, max_retries=10,overgen_factor=0.5):
        self.width = width
        self.height = height
        self.step = step
        self.min_distance = min_distance or (min(width, height) / 20)
        self.target_pieces = target_pieces
        self.min_fusion = min_fusion  # Minimum centroids per final piece
        self.retry_until_success = retry_until_success  # New flag
        self.max_retries = max_retries  # Maximum retry attempts
        
        # Calculate initial Voronoi pieces needed
        if overgen_factor == 0:
            self.initial_pieces = target_pieces
            self.min_fusion = 1
        else:
            self.initial_pieces = int(target_pieces * (min_fusion + overgen_factor))
        
        self.seed_points = []
        self.generation_attempt = 0  # Track attempts
        
        

    def quantize_point(self, point):
        """Snap point to grid"""
        x, y = point
        return (round(x / self.step) * self.step, round(y / self.step) * self.step)
    
    def generate_well_spaced_points(self, k):
        """Generate k points with minimum distance constraint"""
        max_attempts = 1000
        best_points = []
        best_min_distance = 0
        
        # Calculate theoretical maximum for k points using hexagonal packing estimate
        max_possible_distance = np.sqrt(self.width * self.height / k) * 0.9 #Full hex packing factor is ~1.07, reduced for randomness
        
        # If requested distance is too large, warn and reduce
        if self.min_distance > max_possible_distance:
            print(f"Warning: min_distance {self.min_distance:.2f} too large for {k} points.")
            print(f"Reducing to {max_possible_distance:.2f}")
            self.min_distance = max_possible_distance * 0.8
        
        # Try multiple times to find a good configuration
        for attempt in range(max_attempts):
            points = []
            
            # Generate points with distance constraint
            attempts_per_point = 100
            for i in range(k):
                best_candidate = None
                best_distance = 0
                
                for _ in range(attempts_per_point):
                    # Random candidate point
                    x = random.uniform(0.05 * self.width, 0.95 * self.width)
                    y = random.uniform(0.05 * self.height, 0.95 * self.height)
                    candidate = (x, y)
                    
                    # Check distance to existing points
                    if len(points) == 0:
                        best_candidate = candidate
                        best_distance = float('inf')
                        break
                    
                    min_dist_to_existing = min(
                        np.sqrt((candidate[0] - p[0])**2 + (candidate[1] - p[1])**2)
                        for p in points
                    )
                    
                    if min_dist_to_existing >= self.min_distance:
                        best_candidate = candidate
                        best_distance = min_dist_to_existing
                        break
                    elif min_dist_to_existing > best_distance:
                        best_candidate = candidate
                        best_distance = min_dist_to_existing
                
                if best_candidate:
                    points.append(best_candidate)
                else:
                    break
            
            # Check if we got all k points
            if len(points) == k:
                # Calculate minimum distance in this configuration
                min_dist_in_config = float('inf')
                for i, p1 in enumerate(points):
                    for j, p2 in enumerate(points[i+1:], i+1):
                        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        min_dist_in_config = min(min_dist_in_config, dist)
                
                # Keep the best configuration found so far
                if min_dist_in_config > best_min_distance:
                    best_points = points[:]
                    best_min_distance = min_dist_in_config
                    
                    if best_min_distance >= self.min_distance:
                        break
        
        if len(best_points) < k:
            print(f"Warning: Could only place {len(best_points)} points instead of {k}")
        elif best_min_distance < self.min_distance:
            print(f"Warning: Best achieved min_distance: {best_min_distance:.2f} (requested: {self.min_distance:.2f})")
        
        return best_points if best_points else []
    
    def generate_voronoi_pieces(self, k):
        """Generate k Voronoi pieces that tile the rectangle perfectly"""
        
        # Generate well-spaced points
        points = self.generate_well_spaced_points(k)
                
        # Store seed points for visualization
        self.seed_points = points[:]
        
        # Convert to list format for Voronoi
        points_list = [[x, y] for x, y in points]
        
        # Add mirror points outside rectangle to force boundary coverage
        margin = max(self.width, self.height) * 2
        mirror_points = []
        
        # Mirror points below, above, left, right
        for x, y in points_list:
            mirror_points.extend([
                [x, -margin],           # Below
                [x, self.height + margin], # Above
                [-margin, y],           # Left
                [self.width + margin, y]   # Right
            ])
        
        # Corner points
        corner_points = [
            [-margin, -margin],
            [self.width + margin, -margin], 
            [self.width + margin, self.height + margin],
            [-margin, self.height + margin]
        ]
        
        # Combine all points
        all_points = points_list + mirror_points + corner_points
        
        # Create Voronoi diagram
        vor = Voronoi(np.array(all_points))
        
        # Extract pieces for original points only
        pieces = []
        for i in range(len(points_list)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) > 0 and -1 not in region:
                # Get vertices
                vertices = []
                for vertex_idx in region:
                    vertex = vor.vertices[vertex_idx]
                    vertices.append(vertex)
                
                # Clip to rectangle bounds
                clipped_vertices = self.clip_to_rectangle(vertices)
                
                # Quantize to grid
                grid_vertices = [self.quantize_point(v) for v in clipped_vertices]
                
                # Remove duplicates
                clean_vertices = self.remove_duplicates(grid_vertices)
                
                if len(clean_vertices) >= 3:
                    pieces.append(clean_vertices)
        
        return pieces
    
    def clip_to_rectangle(self, vertices):
        """Clip polygon to rectangle using Sutherland-Hodgman algorithm"""
        if not vertices:
            return []
        
        clipped = vertices[:]
        
        # Clip against each edge: left, right, bottom, top
        edges = [
            ('left', 0, lambda p: p[0] >= 0),
            ('right', self.width, lambda p: p[0] <= self.width),
            ('bottom', 0, lambda p: p[1] >= 0),
            ('top', self.height, lambda p: p[1] <= self.height)
        ]
        
        for edge_name, edge_pos, inside_test in edges:
            if not clipped:
                break
                
            input_vertices = clipped[:]
            clipped = []
            
            if len(input_vertices) == 0:
                continue
                
            prev_vertex = input_vertices[-1]
            
            for vertex in input_vertices:
                if inside_test(vertex):
                    if not inside_test(prev_vertex):
                        intersection = self.line_intersection(prev_vertex, vertex, edge_name, edge_pos)
                        if intersection:
                            clipped.append(intersection)
                    clipped.append(vertex)
                elif inside_test(prev_vertex):
                    intersection = self.line_intersection(prev_vertex, vertex, edge_name, edge_pos)
                    if intersection:
                        clipped.append(intersection)
                
                prev_vertex = vertex
        
        # Remove close points
        if len(clipped) > 0:
            cleaned = [clipped[0]]
            for i in range(1, len(clipped)):
                if (abs(clipped[i][0] - cleaned[-1][0]) > 0.001 or 
                    abs(clipped[i][1] - cleaned[-1][1]) > 0.001):
                    cleaned.append(clipped[i])
            
            if (len(cleaned) > 2 and 
                abs(cleaned[-1][0] - cleaned[0][0]) < 0.001 and 
                abs(cleaned[-1][1] - cleaned[0][1]) < 0.001):
                cleaned.pop()
            
            clipped = cleaned
        
        return clipped
    
    def line_intersection(self, p1, p2, edge_name, edge_pos):
        """Find intersection of line segment with rectangle edge"""
        x1, y1 = p1
        x2, y2 = p2
        
        if edge_name in ['left', 'right']:
            if abs(x2 - x1) < 1e-10:
                return None
            t = (edge_pos - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = y1 + t * (y2 - y1)
                return (edge_pos, y)
        
        elif edge_name in ['bottom', 'top']:
            if abs(y2 - y1) < 1e-10:
                return None
            t = (edge_pos - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                return (x, edge_pos)
        
        return None
    
    def remove_duplicates(self, vertices):
        """Remove consecutive duplicate vertices"""
        if len(vertices) <= 1:
            return vertices
        
        clean = [vertices[0]]
        for i in range(1, len(vertices)):
            if vertices[i] != vertices[i-1]:
                clean.append(vertices[i])
        
        if len(clean) > 1 and clean[-1] == clean[0]:
            clean.pop()
        
        return clean
    
    def build_adjacency_graph(self, pieces):
        """Build adjacency graph for MIP formulation"""
        n = len(pieces)
        adjacency = []
        
        # Find all adjacent pairs
        for i in range(n):
            for j in range(i + 1, n):
                if self.pieces_share_edge(pieces[i], pieces[j]):
                    adjacency.append((i, j))
        
        return adjacency
    
    def pieces_share_edge(self, piece1, piece2):
        """Check if two pieces share an edge"""
        # Create edge sets for both pieces
        edges1 = set()
        for i in range(len(piece1)):
            edge = tuple(sorted([piece1[i], piece1[(i + 1) % len(piece1)]]))
            edges1.add(edge)
        
        edges2 = set()
        for i in range(len(piece2)):
            edge = tuple(sorted([piece2[i], piece2[(i + 1) % len(piece2)]]))
            edges2.add(edge)
        
        # Check for shared edges
        return len(edges1.intersection(edges2)) > 0
    
    def solve_mip_partition(self, pieces):
        """Solve the constrained graph partitioning using CPLEX MIP"""
        if not CPLEX_AVAILABLE:
            print("CPLEX not available, falling back to simple fusion")
            return self.fallback_fusion(pieces), False
        
        # NO MIP YET.

    
    def merge_pieces_by_clusters(self, pieces, clusters):
        """Merge pieces according to cluster assignment"""
        merged_pieces = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single piece cluster
                merged_pieces.append(pieces[cluster[0]])
            else:
                # Multi-piece cluster - merge them
                merged_piece = self.merge_multiple_pieces([pieces[i] for i in cluster])
                if merged_piece:
                    merged_pieces.append(merged_piece)
        
        return merged_pieces
    
    def merge_multiple_pieces(self, piece_list):
        """Merge multiple pieces using proper polygon union (remove shared edges)"""
        if len(piece_list) == 1:
            return piece_list[0]
        
        # Start with first piece
        result = piece_list[0][:]
        
        # Successively merge each additional piece
        for i in range(1, len(piece_list)):
            result = self.merge_two_pieces(result, piece_list[i])
            if not result:
                # If merge fails, return first piece
                return piece_list[0]
        
        return result
    
    def merge_two_pieces(self, piece1, piece2):
        """Merge two pieces by removing shared edges"""
        # Find shared edges
        shared_edges = self.find_shared_edges_between_pieces(piece1, piece2)
        
        if not shared_edges:
            # No shared edges - pieces not adjacent, return first piece
            return piece1
        
        # Create edge lists for both pieces
        edges1 = []
        for i in range(len(piece1)):
            start = piece1[i]
            end = piece1[(i + 1) % len(piece1)]
            edges1.append((start, end))
        
        edges2 = []
        for i in range(len(piece2)):
            start = piece2[i]
            end = piece2[(i + 1) % len(piece2)]
            edges2.append((start, end))
        
        # Convert shared edges to set for faster lookup
        shared_edge_set = set()
        for edge in shared_edges:
            shared_edge_set.add((edge[0], edge[1]))
            shared_edge_set.add((edge[1], edge[0]))  # Both directions
        
        # Collect all non-shared edges (the boundary of the union)
        boundary_edges = []
        
        # Add non-shared edges from piece1
        for edge in edges1:
            if edge not in shared_edge_set:
                boundary_edges.append(edge)
        
        # Add non-shared edges from piece2
        for edge in edges2:
            if edge not in shared_edge_set:
                boundary_edges.append(edge)
        
        if len(boundary_edges) < 3:
            return piece1  # Fallback
        
        # Connect edges to form a polygon
        result_polygon = self.connect_edges_to_polygon(boundary_edges)
        
        return result_polygon if len(result_polygon) >= 3 else piece1
    
    def find_shared_edges_between_pieces(self, piece1, piece2):
        """Find shared edges between two pieces"""
        shared_edges = []
        
        # Create edge sets
        edges1 = set()
        for i in range(len(piece1)):
            edge = tuple(sorted([piece1[i], piece1[(i + 1) % len(piece1)]]))
            edges1.add(edge)
        
        edges2 = set()
        for i in range(len(piece2)):
            edge = tuple(sorted([piece2[i], piece2[(i + 1) % len(piece2)]]))
            edges2.add(edge)
        
        # Find intersection
        shared = edges1.intersection(edges2)
        
        # Convert back to edge format
        for edge in shared:
            shared_edges.append(edge)
        
        return shared_edges
    
    def connect_edges_to_polygon(self, edges):
        """Connect a list of edges into a single polygon"""
        if not edges:
            return []
        
        # Start with first edge
        polygon = [edges[0][0], edges[0][1]]
        used_edges = {0}
        
        # Keep connecting edges until we form a closed loop
        while len(used_edges) < len(edges):
            current_end = polygon[-1]
            found_connection = False
            
            # Look for an unused edge that starts at current_end
            for i, (start, end) in enumerate(edges):
                if i in used_edges:
                    continue
                
                if start == current_end:
                    polygon.append(end)
                    used_edges.add(i)
                    found_connection = True
                    break
                elif end == current_end:
                    polygon.append(start)
                    used_edges.add(i)
                    found_connection = True
                    break
            
            if not found_connection:
                # Try to close the polygon
                if polygon[-1] == polygon[0]:
                    polygon.pop()  # Remove duplicate end point
                break
        
        # Remove final point if it's the same as first (closed polygon)
        if len(polygon) > 2 and polygon[-1] == polygon[0]:
            polygon.pop()
        
        return polygon
    
    def fallback_fusion(self, pieces):
        """Fallback method when MIP fails - prioritizes lowest fusing grade pieces"""
        print("Using fallback fusion method with grade prioritization")
        current_pieces = pieces[:]

        # Initialize fusion grades - each piece starts with grade 1 (single original Voronoi cell)
        fusion_grades = [1] * len(current_pieces)

        while len(current_pieces) > self.target_pieces:
            # Find adjacent pairs
            adjacent_pairs = []
            for i in range(len(current_pieces)):
                for j in range(i + 1, len(current_pieces)):
                    if self.pieces_share_edge(current_pieces[i], current_pieces[j]):
                        adjacent_pairs.append((i, j))

            if not adjacent_pairs:
                break
            
            # Find minimum fusion grade among all pieces
            min_grade = min(fusion_grades)

            # Filter pairs where at least one piece has the minimum grade
            min_grade_pairs = []
            for i, j in adjacent_pairs:
                if fusion_grades[i] == min_grade or fusion_grades[j] == min_grade:
                    min_grade_pairs.append((i, j))

            # If no pairs with minimum grade pieces, fall back to any adjacent pair
            if not min_grade_pairs:
                min_grade_pairs = adjacent_pairs

            # Among the minimum grade pairs, prioritize pairs where BOTH pieces have low grades
            # Sort by sum of grades (ascending), then pick randomly from the lowest sum group
            pair_grades = []
            for i, j in min_grade_pairs:
                grade_sum = fusion_grades[i] + fusion_grades[j]
                pair_grades.append((grade_sum, i, j))

            # Sort by grade sum
            pair_grades.sort(key=lambda x: x[0])

            # Find all pairs with the minimum grade sum
            min_sum = pair_grades[0][0]
            best_pairs = [(i, j) for grade_sum, i, j in pair_grades if grade_sum == min_sum]

            # Pick random pair from the best candidates
            i, j = random.choice(best_pairs)
            if DEBUG:
                print(f"Fusing pieces {i} (grade {fusion_grades[i]}) and {j} (grade {fusion_grades[j]}) -> grade {fusion_grades[i] + fusion_grades[j]}")

            # Merge the pieces
            fused = self.merge_two_pieces(current_pieces[i], current_pieces[j])

            # Calculate new fusion grade (sum of the two pieces being fused)
            new_grade = fusion_grades[i] + fusion_grades[j]

            # Replace with fused piece - need to be careful with indices
            # Remove the higher index first to avoid index shifting issues
            higher_idx = max(i, j)
            lower_idx = min(i, j)

            new_pieces = []
            new_grades = []

            for idx in range(len(current_pieces)):
                if idx != higher_idx and idx != lower_idx:
                    new_pieces.append(current_pieces[idx])
                    new_grades.append(fusion_grades[idx])

            # Add the fused piece
            new_pieces.append(fused)
            new_grades.append(new_grade)

            current_pieces = new_pieces
            fusion_grades = new_grades

            # Debug info
            if DEBUG:
                grade_distribution = {}
                for grade in fusion_grades:
                    grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
                print(f"Current pieces: {len(current_pieces)}, Grade distribution: {dict(sorted(grade_distribution.items()))}")

        print(f"Final fusion grades: {sorted(fusion_grades)}")
        return current_pieces
    
    def generate_puzzle_with_retry(self):
        """Generate puzzle with retry logic until MIP succeeds"""
        self.generation_attempt = 0
        
        if not self.retry_until_success:
            # Original behavior - single attempt
            return self.generate_puzzle()
        
        if not CPLEX_AVAILABLE:
            print("Warning: CPLEX not available. Cannot retry for MIP success.")
            return self.generate_puzzle()
        
        print(f"Retry mode enabled - will attempt up to {self.max_retries} times until MIP succeeds")
        print("=" * 70)
        
        for attempt in range(1, self.max_retries + 1):
            self.generation_attempt = attempt
            print(f"\n Attempt {attempt}/{self.max_retries}:")
            
            # Generate initial Voronoi pieces
            pieces = self.generate_voronoi_pieces(self.initial_pieces)
            print(f"Generated {len(pieces)} initial Voronoi pieces")
            
            # Try to solve MIP
            final_pieces, mip_success = self.solve_mip_partition(pieces)
            
            if mip_success:
                print(f" SUCCESS on attempt {attempt}! MIP solver found optimal solution.")
                print(f"   Final result: {len(final_pieces)} pieces (target: {self.target_pieces})")
                return final_pieces
            else:
                print(f" MIP failed on attempt {attempt}")
                if attempt < self.max_retries:
                    print(f"Retrying with new Voronoi configuration...")
        
        print(f"\n Maximum retries ({self.max_retries}) reached without MIP success.")
        print("   Returning result from fallback fusion method.")
        
        # Final attempt with fallback
        pieces = self.generate_voronoi_pieces(self.initial_pieces)
        final_pieces, _ = self.solve_mip_partition(pieces)
        return final_pieces
    
    def generate_puzzle(self):
        """Generate instance using MIP-based constrained graph partitioning"""
        # Generate initial Voronoi pieces
        pieces = self.generate_voronoi_pieces(self.initial_pieces)
        
        print(f"Generated {len(pieces)} initial Voronoi pieces")
        
        # Solve MIP to partition into target number of pieces
        final_pieces, _ = self.solve_mip_partition(pieces)
        
        return final_pieces
    
    def calculate_coverage(self, pieces):
        """Calculate coverage percentage"""
        total_area = sum(self.polygon_area(piece) for piece in pieces)
        rectangle_area = self.width * self.height
        return (total_area / rectangle_area) * 100
    
    def polygon_area(self, vertices):
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0
        
        area = 0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        return abs(area) / 2
    


class BrokenGlassGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Broken Glass Generator")
        self.root.geometry("1200x800")
        self.advanced_visible = False  # start collapsed

        # Parameters
        self.params = {
            'length': tk.DoubleVar(value=25.0),
            'width': tk.DoubleVar(value=20.0),
            'step': tk.DoubleVar(value=1.0),
            'target_pieces': tk.IntVar(value=10),
            'min_distance': tk.DoubleVar(value=1.0),
            'overgen_factor': tk.DoubleVar(value=0.5),
            'length_board': tk.DoubleVar(value=30.0),
            'width_board': tk.DoubleVar(value=25.0),
            'show_centroids': tk.BooleanVar(value=False),
            'seed': tk.StringVar(value=None),
        }
        
        self.generator = None
        self.puzzle_pieces = None
        
        # Create UI
        self.create_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Bind parameter changes to validation
        for var in self.params.values():
            var.trace('w', self.on_parameter_change)
        
        # Initial validation
        self.validate_parameters()
    
    def create_ui(self):
        """Create the main UI layout"""
        # Create main frames
        left_frame = ttk.Frame(self.root, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create parameter controls
        self.create_parameter_controls(left_frame)
        
        # Create matplotlib canvas
        self.create_matplotlib_canvas(right_frame)

        self.root.after(0, self.bring_to_front)

    
    def bring_to_front(self):
        window = self.root
        # Raise the window
        window.lift()
        # Make it topmost for a moment
        window.attributes('-topmost', True)
        # Then allow other windows to go on top again
        window.after_idle(window.attributes, '-topmost', False)
        # Try to grab focus
        window.focus_force()
        
    def run_nestab(self,dataset,instance,outdir):

        def bundled_path(rel):
            base = getattr(sys, "_MEIPASS", os.path.abspath("."))
            return os.path.join(base, rel)

        tool = bundled_path(os.path.join("bin", "tool.exe" if sys.platform.startswith("win") else "tool"))

        # (optional) ensure executable bit on Linux/macOS
        if not sys.platform.startswith("win"):
            try: os.chmod(tool, 0o755)
            except OSError: pass

        result = subprocess.run([tool, "--help"], capture_output=True, text=True, check=True)
        print(result.stdout)
    
    

    def create_parameter_controls(self, parent):
        """Create parameter control widgets"""
        
        
        
        # Title
        title_label = ttk.Label(parent, text="Parameters", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        
        # Parameter controls
        group_frame1 = self.create_slider_group(parent, "Strip Size", [
            ("Board Heigth(H):", 'length_board', 5, 60,1),
            ("Board Width(W):", 'width_board', 5, 50,1),
        ])
        slider_frame = ttk.Frame(group_frame1)
        slider_frame.pack(fill=tk.X, pady=2)
        

        self.create_slider_group(parent, "Glass Size", [
            ("Heigth:", 'length', 5, 50,1),
            ("Width:", 'width', 5, 40,1),
            ("Number of Polygons:", 'target_pieces', 5, 100,1)
        ])
        
        self.advanced_button = ttk.Button(
            parent,
            text="Show Advanced Settings ▾",
            command=self.toggle_advanced
        )
        self.advanced_button.pack(fill="x", padx=10, pady=(0, 5))
        self.advanced_frame = ttk.Frame(parent, padding=10, relief="groove")
        
        self.create_slider_group(self.advanced_frame, "Generation Parameters", [
            ("Min Distance:", 'min_distance', 0.1, 5.0,0.1),
            ("Polygon Irregularity:", 'overgen_factor', 0, 2.0,0.1),
            ("Grid Step:", 'step', 0.1, 2.0,0.1),
        ])
        ttk.Label(self.advanced_frame, text="Seed", width=15).pack(side=tk.LEFT)
        ttk.Entry(self.advanced_frame, textvariable=self.params['seed'], width=15).pack(side=tk.RIGHT)
         # Label
        tk.Label(self.advanced_frame, text="").pack(pady=5)
        ttk.Checkbutton(self.advanced_frame, text="Show Centroids", variable=self.params['show_centroids']).pack(anchor=tk.W, pady=10)
       
        # Validation status
        self.status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(self.status_frame, text="All parameters valid", 
                                    foreground="green")
        self.status_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)
    
        self.generate_btn = ttk.Button(button_frame, text="Generate Instance", 
                                     command=self.generate_puzzle)
        self.generate_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_parameters).pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="Save polygons", 
                  command=self.save_instance).pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="Generate graph", 
                  command=self.save_instance_generate_graph).pack(fill=tk.X, pady=2)
    

    def toggle_advanced(self):
        """Show or hide the advanced_frame."""
        if self.advanced_visible:
            # Hide it
            self.advanced_frame.pack_forget()
            self.advanced_button.config(text="Show Advanced Settings ▾")
            self.advanced_visible = False
        else:
            # Show it
            self.advanced_frame.pack(fill="x", padx=10, pady=(0, 10),after=self.advanced_button)
            self.advanced_button.config(text="Hide Advanced Settings ▴")
            self.advanced_visible = True

    def create_slider_group(self, parent, title, sliders):
        """Create a group of parameter sliders with editable value fields"""
        group_frame = ttk.LabelFrame(parent, text=title, padding=10)
        group_frame.pack(fill=tk.X, pady=5)

        # Callback for Resolution
        def make_discrete_callback(var, resolution):
            def callback(value):
                # Round to nearest discrete step
                discrete_value = round(float(value) / resolution) * resolution
                var.set(discrete_value)
            return callback
        
        # Parameters that should have integer input fields for values
        int_field_params = ['target_pieces', 'length', 'width', 'length_board', 'width_board']
        
        
        for label, param_key, min_val, max_val,res in sliders:
            slider_frame = ttk.Frame(group_frame)
            slider_frame.pack(fill=tk.X, pady=2)
            
            # Label
            ttk.Label(slider_frame, text=label, width=20).pack(side=tk.LEFT)
            
            # Slider
            if isinstance(self.params[param_key], tk.IntVar):
                slider = ttk.Scale(slider_frame, from_=min_val, to=max_val,variable=self.params[param_key], orient=tk.HORIZONTAL,
                                   command=make_discrete_callback(self.params[param_key], res))
            else:
                slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, 
                                 variable=self.params[param_key], orient=tk.HORIZONTAL,length=150,
                                 command=make_discrete_callback(self.params[param_key], res)
                                 )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
            
            # Value display - either editable entry or read-only label
            if param_key in int_field_params:
                # Create editable integer entry field
                value_entry = ttk.Entry(slider_frame, width=6, justify='center')
                value_entry.pack(side=tk.RIGHT)
                
                # Set initial value
                value_entry.insert(0, str(int(self.params[param_key].get())))
                
                # Update entry when slider changes
                def update_entry_from_slider(param=param_key, entry=value_entry):
                    current_text = entry.get()
                    new_value = str(int(self.params[param].get()))
                    if current_text != new_value:
                        entry.delete(0, tk.END)
                        entry.insert(0, new_value)
                        entry.config(foreground='black')
                
                # Update slider when entry changes
                def update_slider_from_entry(event=None, param=param_key, entry=value_entry):
                    try:
                        value = int(entry.get())
                        if min_val <= value:
                            self.params[param].set(value)
                            entry.config(foreground='black')
                        else:
                            print("out of range min ",min_val," max: ",max_val, " current val: ",value)
                            entry.config(foreground='red')
                    except ValueError:
                        print("value error!")
                        entry.config(foreground='red')
                
                # Bind events
                self.params[param_key].trace('w', lambda *args, update=update_entry_from_slider: update())
                value_entry.bind('<KeyRelease>', update_slider_from_entry)
                value_entry.bind('<FocusOut>', update_slider_from_entry)
                value_entry.bind('<Return>', update_slider_from_entry)
                
            else:
                # Create read-only value display for continuous parameters
                value_label = ttk.Label(slider_frame, text=f"{self.params[param_key].get():.1f}", width=6)
                value_label.pack(side=tk.RIGHT)
                
                # Update value display when slider changes
                def update_display(param=param_key, label=value_label):
                    if isinstance(self.params[param], tk.IntVar):
                        label.config(text=f"{self.params[param].get()}")
                    else:
                        label.config(text=f"{self.params[param].get():.1f}")
                
                self.params[param_key].trace('w', lambda *args, update=update_display: update())
        return group_frame
    
    def create_matplotlib_canvas(self, parent):
        """Create matplotlib canvas for puzzle display"""
        # Create figure with equal subplot heights - FIXED: added height_ratios
        self.fig, (self.ax_preview, self.ax_result) = plt.subplots(2, 1, figsize=(8, 10), 
                                                                   gridspec_kw={'height_ratios': [1, 1]})
        self.fig.suptitle(' ', fontsize=14, fontweight='bold')

        self.ax_result.set_aspect('equal')
        # FIXED: Proper spacing that doesn't cause title overlap
        self.fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
        # Initial preview
        self.update_preview()
    
    def on_parameter_change(self, *args):
        """Called when any parameter changes"""
        self.validate_parameters()
        self.update_preview()
    
    def validate_parameters(self):
        """Validate parameters and update status"""
        errors = []
        warnings = []
        
        # Get current values
        length = self.params['length'].get()
        width = self.params['width'].get()
        length_board = self.params['length_board'].get()
        width_board = self.params['width_board'].get()
        min_distance = self.params['min_distance'].get()
        target_pieces = self.params['target_pieces'].get()
        step = self.params['step'].get()
        

        # Check board constraints
        if length_board < length:
            errors.append(f"Board length ({length_board:.1f}) must be >= length ({length:.1f})")
        
        if width_board < width:
            errors.append(f"Board width ({width_board:.1f}) must be >= width ({width:.1f})")
        
        # Check logical constraints
        if min_distance > min(length, width) / 4:
            warnings.append("Min distance might be too large for puzzle dimensions")
        
        if target_pieces > length * width / 4:
            warnings.append("Target pieces might be too many for puzzle size")
        
        if step > min(length, width) / 10:
            warnings.append("Step size might be too large")
        
        # Update status
        if len(errors)>0:
            status_text = "ERRORS:\n" + "\n".join(errors[:2])  # Limit to first 2 errors
        else:
            status_text = ""
        
        self.status_label.config(text=status_text, foreground="red")
        self.generate_btn.config(state="normal")


    
    def update_preview(self):
        """Update the preview plot"""
        self.ax_preview.clear()
        self.ax_preview.set_title('Parameter Preview', fontweight='bold')
        
        # FIXED: Clear any existing preview info text
        if hasattr(self, 'preview_info_text'):
            self.preview_info_text.remove()
        
        # Get current values
        length = self.params['length'].get()
        width = self.params['width'].get()
        length_board = self.params['length_board'].get()
        width_board = self.params['width_board'].get()
        step = self.params['step'].get()
        target_pieces = self.params['target_pieces'].get()
        
        # Draw puzzle area
        puzzle_rect = plt.Rectangle((0, 0), length, width, 
                                  fill=False, edgecolor='blue', linestyle='--',alpha=0.7,linewidth=2, label='Glass Area')
        self.ax_preview.add_patch(puzzle_rect)
        
        # Draw board area (if different)
        if length_board != length or width_board != width:
            board_rect = plt.Rectangle((0, 0), length_board, width_board, 
                                     fill=False, edgecolor='red', linewidth=2,label='Board Area')
            self.ax_preview.add_patch(board_rect)
        
        # Show grid (simplified - just major lines)
        if step <= 2.0:
            # Draw major grid lines only
            self.ax_preview.grid(True, alpha=0.3)
        
        # Show approximate piece size
        approx_area_per_piece = (length * width) / target_pieces
        approx_piece_size = np.sqrt(approx_area_per_piece)
        
        sample_rect = plt.Rectangle((length * 0.05, width * 0.05), 
                                  approx_piece_size, approx_piece_size,
                                  fill=True, facecolor='yellow', alpha=0.5, 
                                  edgecolor='orange', linewidth=1,
                                  label=f'~Piece Size ({approx_piece_size:.1f}²)')
        self.ax_preview.add_patch(sample_rect)
        


        # Set limits and formatting
        max_dim = max(length_board, width_board)
        self.ax_preview.set_xlim(-max_dim * 0.1, max_dim * 1.1)
        self.ax_preview.set_ylim(-max_dim * 0.1, max_dim * 1.1)
        self.ax_preview.set_aspect('equal')
        
        # Place legend outside the plot area (to the right)
        self.ax_preview.legend(bbox_to_anchor=(0.09, 0.80),borderaxespad=0.0, borderpad=0.3,loc='upper left',bbox_transform=self.fig.transFigure, fontsize=8)
        
        # FIXED: Add info text in the space between plots (external to plot area)
        info_text = f'Puzzle: {length:.0f}×{width:.0f}\nBoard: {length_board:.0f}×{width_board:.0f}\nTarget: {target_pieces} pieces \nGrid: {step:.1f}'
        self.preview_info_text = self.fig.text(0.1, 0.90, info_text, ha='left', va='center', fontsize=9,
                                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # Only redraw if canvas exists
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
    
    def setup_result_placeholder(self):
        """Setup placeholder content for the result area"""
        self.ax_result.clear()
        self.ax_result.set_title('Generated Instance', fontweight='bold')
        
        # Draw placeholder content to show the area
        self.ax_result.text(0.5, 0.5, 'Click "Generate Puzzle" to see results here', 
                           transform=self.ax_result.transAxes,
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Set equal aspect and limits similar to preview
        self.ax_result.set_xlim(-2, 27)
        self.ax_result.set_ylim(-2, 22)
        self.ax_result.set_aspect('equal')
        self.ax_result.grid(True, alpha=0.3)
        
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
    
    def generate_puzzle(self):
        """Generate the jigsaw puzzle with current parameters"""
        # Disable button during generation
        self.generate_btn.config(state="disabled", text="Generating...")
        self.root.update()
        
        try:
            # Get parameters
            params = {key: var.get() for key, var in self.params.items()}
            seed_string = params['seed']
            seed = None
            print("Using seed:", seed_string)
            if seed_string is not None and seed_string != '':
                try:
                    seed = int(seed_string)
                except ValueError:
                    try:
                        print("trying seed string to ascii number conversion")
                        Ascii_seed = ""
                        for w in seed_string:
                            Ascii_seed += str(ord(w))
                        seed = int(Ascii_seed) % int(sys.maxsize)
                        #seed = int(Ascii_seed)
 
                        print("Converted seed string to integer:", seed)
                    except ValueError:
                        print("Invalid seed value, using random seed")
                        seed = None
            random.seed(seed)
            # Create generator
            self.generator = GridJigsawGenerator(
                width=params['length'],
                height=params['width'],
                step=params['step'],
                min_distance=params['min_distance'],
                target_pieces=params['target_pieces'],
                min_fusion=2,
                retry_until_success=True,
                max_retries=8,
                overgen_factor=params['overgen_factor']
            )
            
            # Show progress
            self.ax_result.clear()
            self.ax_result.set_title('Generating...', fontweight='bold')
            self.ax_result.text(0.5, 0.5, 'Please wait...', 
                              transform=self.ax_result.transAxes,
                              ha='center', va='center', fontsize=14)
            self.canvas.draw_idle()
            self.root.update()
            
            # Generate puzzle
            print("Generating puzzle with parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            print()
            
            self.puzzle_pieces = self.generator.generate_puzzle_with_retry()
            
            # Display result
            self.display_result()
            
        except Exception as e:
            # Show error
            self.ax_result.clear()
            self.ax_result.set_title('Generation Failed', fontweight='bold', color='red')
            self.ax_result.text(0.5, 0.5, f'Error: {str(e)}', 
                              transform=self.ax_result.transAxes,
                              ha='center', va='center', fontsize=12, color='red')
            self.canvas.draw_idle()
            messagebox.showerror("Generation Error", f"Failed to generate puzzle:\n{str(e)}")
        
        finally:
            # Re-enable button
            self.generate_btn.config(state="normal", text="Generate Puzzle")
    
    def display_result(self):
        show_centroids = self.params['show_centroids'].get()
        """Display the generated instance result"""
        if not self.puzzle_pieces or not self.generator:
            return
        
        self.ax_result.clear()
        
        # FIXED: Clear any existing result info text
        if hasattr(self, 'result_info_text'):
            self.result_info_text.remove()
        
        # Draw pieces
        colors = cm.Set3(np.linspace(0, 1, len(self.puzzle_pieces)))
        for i, piece in enumerate(self.puzzle_pieces):
            if len(piece) >= 3:
                polygon = Polygon(piece, closed=True, 
                                facecolor=colors[i],
                                edgecolor='black', linewidth=1.5, alpha=0.8)
                self.ax_result.add_patch(polygon)
        
        # Show centroids
        if hasattr(self.generator, 'seed_points') and show_centroids:
            for cx, cy in self.generator.seed_points:
                self.ax_result.plot(cx, cy, 'ko', markersize=3, markerfacecolor='red', 
                                  markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        
        # Rectangle boundary
        length = self.params['length'].get()
        width = self.params['width'].get()
        boundary = plt.Rectangle((0, 0), length, width, 
                               fill=False, edgecolor='red', linewidth=2)
        self.ax_result.add_patch(boundary)
        
        # Format
        self.ax_result.set_xlim(-1, length + 1)
        self.ax_result.set_ylim(-1, width + 1)
        self.ax_result.set_aspect('equal')
        self.ax_result.set_title('Generated Instance', fontweight='bold')
        
        # Info
        coverage = self.generator.calculate_coverage(self.puzzle_pieces)
        target_pieces = self.params['target_pieces'].get()
        info_text = f'Generated: {len(self.puzzle_pieces)} pieces\n Coverage: {coverage:.1f}%\n'
        if hasattr(self.generator, 'generation_attempt') and self.generator.generation_attempt > 0:
            info_text += f'Attempt: {self.generator.generation_attempt}'
        
        # FIXED: Place info text in the bottom margin area (external to plot area)
        self.result_info_text = self.fig.text(0.1, 0.53, info_text, ha='left', va='center', fontsize=9,
                                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        self.canvas.draw()
    
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        defaults = {
            'length': 25.0,
            'width': 20.0,
            'step': 1.0,
            'target_pieces': 10,
            'min_distance': 1.0,
            'overgen_factor': 0.5,
            'length_board': 30.0,
            'width_board': 25.0
        }
        
        for key, value in defaults.items():
            self.params[key].set(value)
    
    def save_instance(self):
        """Save the current instance"""
        
        if self.puzzle_pieces:
            params = {key: var.get() for key, var in self.params.items()}
            finalJson = {}
            piece_num = 1
            pieceCopy = copy.deepcopy(self.puzzle_pieces)
            for piece in pieceCopy:
                left_bottom = (9999,9999)
                pivot_index = 0

                for i in range(0,len(piece)):
                    if piece[i][0] < left_bottom[0] or piece[i][1] < left_bottom[1]:
                        left_bottom = piece[i]
                        pivot_index = i
                
                for i in range(0,len(piece)):
                    piece[i] = {"x": piece[i][0] - left_bottom[0], "y":piece[i][1] - left_bottom[1]}
                #reorder the piece so the first is always the (0,0)
                piece = piece[pivot_index:]+piece[:pivot_index]

                finalJson[("piece "+str(piece_num))] = {
                    "VERTICES":piece,
                    "QUANTITY": 1,
                    "NUMBER OF VERTICES": len(piece)
                }
                piece_num += 1

            finalJson["rect"] = [
                {"x":0,"y":0},
                {"x":0,"y":params['width_board']},
                {"x":params['length_board'],"y":params['width_board']},
                {"x":params['length_board'],"y":0}
            ]

            script_dir = BASE_DIR

            file_path = filedialog.asksaveasfilename(
                initialdir=script_dir,
                initialfile="jigsawPacking.json",  # Suggest a default filename
                title="Save file as...",
                defaultextension=".json",
                filetypes=[
                    ("Json", "*.json")
                ]
            )

            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        json.dump(finalJson, f, indent=4, ensure_ascii=False)

                    metadata_path = Path(file_path).with_suffix('.txt')
                    with open(metadata_path, 'w') as f:
                        f.write("Optimal Length:\t"+str(params['length']))
                    print(f"File saved: {file_path}")
                except Exception as e:
                    print(f"Error: {e}")
                else:
                    messagebox.showinfo("Success", f"JSON saved successfully!\n{file_path}")
        else:
            messagebox.showinfo("Error","No polygon set detcted, please generate instance first.")
    
    

    def save_instance_generate_graph(self):
        """Save the current instance"""
        
        if self.puzzle_pieces:
            script_dir = BASE_DIR

            file_path = Path(filedialog.askdirectory(
                initialdir=script_dir,
                title="choose an empty folder to save..."
            ))
        
            params = {key: var.get() for key, var in self.params.items()}
            finalJson = {}
            piece_num = 1
            pieceCopy = copy.deepcopy(self.puzzle_pieces)
            for piece in pieceCopy:
                left_bottom = (9999,9999)
                pivot_index = 0
                #print(piece)
                for i in range(0,len(piece)):
                    if piece[i][0] < left_bottom[0] or piece[i][1] < left_bottom[1]:
                        left_bottom = piece[i]
                        pivot_index = i
                        #print(piece[i])
                
                for i in range(0,len(piece)):
                    piece[i] = {"x": piece[i][0] - left_bottom[0], "y":piece[i][1] - left_bottom[1]}
                #reorder the piece so the first is always the (0,0)
                piece = piece[pivot_index:]+piece[:pivot_index]

                finalJson[(str(piece_num))] = {
                    "VERTICES":piece,
                    "QUANTITY": 1,
                    "NUMBER OF VERTICES": len(piece)
                }
                piece_num += 1

                #print(piece)
            
            finalJson["rect"] = [
                {"x":0,"y":0},
                {"x":0,"y":params['width_board']},
                {"x":params['length_board'],"y":params['width_board']},
                {"x":params['length_board'],"y":0}
            ]

            

            if(any(file_path.iterdir())):
                messagebox.showinfo("Error", f"folder not empty!")
                return
            
            instanceJson = [{
                "dataset":file_path.name,
                "outputName":file_path.name,
                "quantity":1,
                "width":params['width_board'],
                "length":params['length'],
                "cuts":[int(params['length'])-1,params['length']]
            }]

            datasetDir = BUNDLE_DIR / "dataset_tmp"
            instanceDir =BUNDLE_DIR / "instance_tmp"
            try:
                if datasetDir.is_dir():
                    shutil.rmtree(datasetDir)
                if instanceDir.is_dir():
                    shutil.rmtree(instanceDir)
            except Exception as e:
                print("fail to remove tmp folder")

            os.makedirs(BUNDLE_DIR / "dataset_tmp",exist_ok=True)
            os.makedirs(BUNDLE_DIR / "instance_tmp",exist_ok=True)
            
            tmpDataset = BUNDLE_DIR / "dataset_tmp" / (file_path.name+".json")
            try:
                with open(tmpDataset, 'w') as f:
                    json.dump(finalJson, f, indent=4, ensure_ascii=False)
                print(f"File saved: {tmpDataset}")
            except Exception as e:
                print(f"Error: {e}")
                messagebox.showinfo("Error", f"JSON failed to save!\n{tmpDataset}")
            
            tmpInstance = BUNDLE_DIR / "instance_tmp" / (file_path.name+".json")
            try:
                with open(tmpInstance, 'w') as f:
                    json.dump(instanceJson, f, indent=4, ensure_ascii=False)
                print(f"File saved: {tmpInstance}")
            except Exception as e:
                messagebox.showinfo("Error", f"failed to generate instance!\n{tmpInstance}")
            
            
            
            args = [
                str(BUNDLE_DIR / "bin" / "nesting_graph.exe"),
                "--datasets",  str(datasetDir),
                "--instances", str(instanceDir),
                "--apply_cuts", "1",
                "--cliqueCovering", "1",
                "--typeOriented", "0",
                "--outputdir", str(file_path),
            ]

            cmdline = subprocess.list2cmdline(args)
            print(cmdline)
            # IMPORTANT: only one outer quote pair
            p = subprocess.Popen(
                ["cmd.exe", "/v:on", "/k", f'"{cmdline}"'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            rc = p.wait()
            print("exit code:", rc)

            if(str(rc) != '0'):
                messagebox.showinfo("Cancel","Generation interrupted.")
            else:
                messagebox.showinfo("OK","Generation complete.")
            print("graph generation complete")
            try:
                shutil.rmtree(datasetDir)
                shutil.rmtree(instanceDir)
            except Exception as e:
                print("fail to remove tmp folder..")
            
            #messagebox.showinfo("Success","graph is sucessfully generated.")
        else:
            messagebox.showinfo("Error","No polygon set detcted, please generate instance first.")
    


    def on_closing(self):
        """Handle window closing event to ensure proper cleanup"""
        try:
            # Close matplotlib figures
            plt.close('all')
            
            # Destroy the tkinter window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Force exit the Python process
            import sys
            sys.exit(0)



def main():
    """Main application entry point"""
    print("Starting Tkinter + Matplotlib Jigsaw Puzzle Generator")
    print("=" * 60)
    
    #if not CPLEX_AVAILABLE:
    #    print("Note: CPLEX not detected. Using fallback fusion method.")
    #    print()
    
    root = tk.Tk()
    app = BrokenGlassGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()