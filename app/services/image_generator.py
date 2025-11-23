"""
Image Generation Service for Meetara Core.

Generates mathematical diagrams, graphs, and geometric visualizations dynamically.
"""
import re
import io
import base64
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
import numpy as np
from app.core.logger import api_logger


class ImageGenerator:
    """Service for generating mathematical and geometric diagrams."""
    
    def __init__(self):
        """Initialize the image generator."""
        self.output_dir = Path("images/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Image generator initialized. Output directory: {self.output_dir}")
    
    def should_generate_image(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Minimal fallback method - ONLY used when NO RAG context is available.
        
        ⚠️ This is keyword-based and should be avoided. Use should_generate_image_from_rag()
        instead which uses RAG context and LLM analysis (like text generation).
        
        This method is kept as a last resort for edge cases with no documents retrieved.
        
        Returns:
            Tuple[bool, Optional[str]]: (should_generate, image_type)
        """
        query_lower = query.lower()
        
        # ✅ ONLY check for very explicit requests when no RAG is available
        # Require explicit image request keyword + specific concept
        explicit_image_request = any(kw in query_lower for kw in ["graph", "plot", "diagram", "draw", "show me a", "visualize"])
        
        if not explicit_image_request:
            return False, None
        
        # Very explicit function notation (e.g., "graph of y = x²")
        if re.search(r'[yf]\(?[x]\)?\s*=|y\s*=\s*[^=]+', query_lower):
            return True, "function_graph"
        
        # Very explicit requests with domain keywords (minimal matching)
        if "triangle" in query_lower or "angle" in query_lower:
            return True, "geometry"
        
        if "integral" in query_lower or "derivative" in query_lower:
            return True, "calculus_diagram"
        
        # Only return True if VERY explicit (this prevents false positives)
        # Without RAG context, we can't be smart about it - so be conservative
        return False, None
    
    def generate_function_graph(self, expression: str, title: str = None) -> Optional[str]:
        """
        Generate a graph of a mathematical function.
        
        Args:
            expression: Function expression (e.g., "x^2", "sin(x)", "x**2")
            title: Optional title for the graph
            
        Returns:
            Path to generated image file or None if failed
        """
        try:
            # Parse common function expressions
            x = np.linspace(-10, 10, 1000)
            
            # Convert expression to Python code
            expr_code = self._parse_function(expression)
            
            # Evaluate function
            try:
                y = eval(expr_code)
            except:
                # Fallback for simple expressions
                if "sin" in expression.lower():
                    y = np.sin(x)
                elif "cos" in expression.lower():
                    y = np.cos(x)
                elif "tan" in expression.lower():
                    y = np.tan(x)
                elif "log" in expression.lower() or "ln" in expression.lower():
                    y = np.log(np.abs(x) + 0.001)
                elif "exp" in expression.lower() or "e^" in expression.lower():
                    y = np.exp(x/3)  # Scale for visibility
                elif "sqrt" in expression.lower() or "√" in expression:
                    y = np.sqrt(np.maximum(x, 0))
                elif "^2" in expression or "**2" in expression or "x²" in expression:
                    y = x ** 2
                elif "^3" in expression or "**3" in expression:
                    y = x ** 3
                else:
                    # Default to linear
                    y = x
            else:
                # Clip very large values for visibility
                y = np.clip(y, -50, 50)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, linewidth=2, color='#2563eb')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title(title or f'Graph of {expression}', fontsize=14, fontweight='bold')
            
            # Save image
            filename = f"generated_function_{hash(expression) % 10000}.png"
            filepath = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            api_logger.info(f"Generated function graph: {filepath}")
            return str(filepath.relative_to(Path("images")))
            
        except Exception as e:
            api_logger.error(f"Error generating function graph: {e}")
            return None
    
    def generate_geometry_diagram(self, shape_type: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Generate a geometric diagram.
        
        Args:
            shape_type: Type of shape (triangle, circle, rectangle, etc.)
            params: Parameters for the shape
            
        Returns:
            Path to generated image file or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            ax.axis('off')
            
            if shape_type == "triangle":
                # Extract triangle parameters
                angles = params.get("angles", [60, 60, 60])
                vertices = self._calculate_triangle_vertices(angles, params.get("side_lengths"))
                
                triangle = Polygon(vertices, fill=True, alpha=0.3, color='#3b82f6', edgecolor='#1e40af', linewidth=2)
                ax.add_patch(triangle)
                
                # Label vertices and angles
                labels = params.get("labels", ["A", "B", "C"])
                for i, (v, label) in enumerate(zip(vertices, labels)):
                    ax.text(v[0], v[1], label, fontsize=14, fontweight='bold', ha='center', va='center')
                    # Angle labels
                    if angles[i] is not None:
                        ax.text(v[0] + 0.3, v[1] + 0.3, f"{angles[i]}°", fontsize=10, color='#dc2626')
                
                ax.set_xlim(-2, 6)
                ax.set_ylim(-1, 5)
                title = params.get("title", "Triangle")
                
            elif shape_type == "circle":
                radius = params.get("radius", 2)
                center = params.get("center", [0, 0])
                
                circle = Circle(center, radius, fill=True, alpha=0.3, color='#3b82f6', edgecolor='#1e40af', linewidth=2)
                ax.add_patch(circle)
                
                # Draw radius
                ax.plot([center[0], center[0] + radius], [center[1], center[1]], 'r--', linewidth=1.5)
                ax.text(center[0] + radius/2, center[1] - 0.3, f'r = {radius}', fontsize=10, color='red')
                
                ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
                ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
                title = params.get("title", "Circle")
                
            else:
                # Default: rectangle
                width = params.get("width", 4)
                height = params.get("height", 3)
                rect = Rectangle((0, 0), width, height, fill=True, alpha=0.3, color='#3b82f6', edgecolor='#1e40af', linewidth=2)
                ax.add_patch(rect)
                ax.set_xlim(-0.5, width + 0.5)
                ax.set_ylim(-0.5, height + 0.5)
                title = params.get("title", "Rectangle")
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save image
            filename = f"generated_geometry_{shape_type}_{hash(str(params)) % 10000}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            api_logger.info(f"Generated geometry diagram: {filepath}")
            return str(filepath.relative_to(Path("images")))
            
        except Exception as e:
            api_logger.error(f"Error generating geometry diagram: {e}")
            return None
    
    def _parse_function(self, expression: str) -> str:
        """Parse a function expression into Python code."""
        expr = expression.lower().replace(" ", "")
        
        # Replace common math notation
        expr = expr.replace("^", "**")
        expr = expr.replace("²", "**2")
        expr = expr.replace("³", "**3")
        
        # Handle trigonometric functions
        expr = re.sub(r'sin\(([^)]+)\)', r'np.sin(\1)', expr)
        expr = re.sub(r'cos\(([^)]+)\)', r'np.cos(\1)', expr)
        expr = re.sub(r'tan\(([^)]+)\)', r'np.tan(\1)', expr)
        
        # Ensure x variable is used
        if "x" not in expr:
            expr = "x" if not expr else expr.replace("y", "x")
        
        return expr
    
    def _calculate_triangle_vertices(self, angles: list, side_lengths: Optional[list] = None):
        """Calculate triangle vertices from angles or side lengths."""
        if side_lengths:
            # Use side lengths if provided
            a, b, c = side_lengths[:3]
            # Place first vertex at origin, second on x-axis
            vertices = np.array([[0, 0], [a, 0], [0, 0]])
            # Calculate third vertex using law of cosines
            angle_C = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
            vertices[2] = [b * np.cos(angle_C), b * np.sin(angle_C)]
        else:
            # Use angles (approximate equilateral for default)
            vertices = np.array([[0, 0], [4, 0], [2, 3.5]])
        
        return vertices
    
    def should_generate_image_from_rag(self, query: str, retrieved_docs: List[Any], domain: str = "academic_tutoring", llm_processor=None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Intelligently determine image generation needs based on RAG-retrieved context.
        
        Uses LLM to analyze retrieved documents (just like text generation) to determine
        what type of diagram would be helpful. This is context-aware, not keyword-based.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved from RAG
            domain: Domain context
            llm_processor: Optional LLM processor for intelligent analysis
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (should_generate, image_type, reasoning)
        """
        if not retrieved_docs or len(retrieved_docs) == 0:
            # No RAG context - only generate if query explicitly requests image
            query_lower = query.lower()
            explicit_image_requests = ["graph", "plot", "diagram", "draw", "show", "visualize"]
            if any(kw in query_lower for kw in explicit_image_requests):
                should_gen, img_type = self.should_generate_image(query)
                return should_gen, img_type, "explicit_request_no_rag"
            return False, None, "no_rag_context"
        
        # ✅ RAG-BASED ANALYSIS: Analyze retrieved document content to determine image needs
        # Combine top retrieved documents (same approach as text generation)
        doc_contents = []
        for doc in retrieved_docs[:3]:  # Top 3 most relevant
            content = doc.page_content.strip()[:800]  # Limit per doc
            if content:
                doc_contents.append(content)
        
        if not doc_contents:
            return False, None, "no_retrieved_content"
        
        combined_context = "\n\n".join(doc_contents)
        combined_lower = combined_context.lower()
        query_lower = query.lower()
        
        # ✅ PRIORITY 1: Explicit image requests in query (user wants a diagram)
        explicit_request = any(kw in query_lower for kw in ["graph", "plot", "diagram", "draw", "show me a", "visualize"])
        
        # ✅ PRIORITY 2: Use LLM analysis if available (like text generation!)
        if llm_processor:
            try:
                # Ask LLM to analyze if an image would be helpful
                analysis_prompt = f"""Based on the following query and retrieved documents, determine if a diagram/graph would be helpful to explain the concept.

Query: {query}

Retrieved Context:
{combined_context[:1500]}

Answer ONLY with one of these options:
- "function_graph" if it's about mathematical functions, equations like y=x², f(x)=sin(x)
- "geometry" if it's about triangles, circles, angles, geometric shapes
- "calculus_diagram" if it's about integrals, derivatives, limits
- "chemistry_diagram" if it's about molecules, chemical reactions, compounds
- "biology_diagram" if it's about cells, DNA, organisms
- "physics_diagram" if it's about forces, motion, circuits, waves
- "none" if no diagram is needed

Answer:"""
                
                llm_response = llm_processor.generate_response(
                    query=analysis_prompt,
                    context="",
                    domain=domain,
                    use_thinking=False
                )
                llm_answer = llm_response.get("response", "").strip().lower()
                
                # Parse LLM response
                if "function_graph" in llm_answer or "function" in llm_answer:
                    if explicit_request:
                        return True, "function_graph", "llm_analysis"
                elif "geometry" in llm_answer:
                    if explicit_request:
                        return True, "geometry", "llm_analysis"
                elif "calculus" in llm_answer:
                    if explicit_request:
                        return True, "calculus_diagram", "llm_analysis"
                elif "chemistry" in llm_answer:
                    if explicit_request:
                        return True, "chemistry_diagram", "llm_analysis"
                elif "biology" in llm_answer:
                    if explicit_request:
                        return True, "biology_diagram", "llm_analysis"
                elif "physics" in llm_answer:
                    if explicit_request:
                        return True, "physics_diagram", "llm_analysis"
            except Exception as e:
                api_logger.warning(f"LLM image analysis failed: {e}, falling back to context analysis")
        
        # ✅ PRIORITY 3: Minimal context-aware fallback (if LLM unavailable)
        # Only use this if LLM analysis failed - prefer LLM over keywords!
        # This is a safety net, not the primary method
        
        # Only proceed if explicit request AND we have retrieved context
        if not explicit_request:
            return False, None, "no_explicit_request"
        
        # Check for clear mathematical notation in query (most reliable)
        has_function_notation = re.search(r'[yf]\(?[x]\)?\s*=|y\s*=\s*[^=]+', query_lower)
        if has_function_notation:
            return True, "function_graph", "context_analysis_function"
        
        # Minimal pattern matching in RETRIEVED CONTEXT (not just query)
        # This uses what RAG found, not keywords in query
        if "integral" in combined_lower or "∫" in combined_lower or "derivative" in combined_lower:
            return True, "calculus_diagram", "context_analysis"
        
        if "triangle" in combined_lower or "angle" in combined_lower or "geometry" in combined_lower:
            return True, "geometry", "context_analysis"
        
        # Chemistry - ONLY if NO math context (prevents leaks!)
        has_math_in_context = has_function_notation or any(kw in combined_lower for kw in ["graph", "plot", "equation", "y =", "f(x)"])
        if not has_math_in_context:
            if "molecule" in combined_lower or "chemical" in combined_lower:
                return True, "chemistry_diagram", "context_analysis"
        
        if "cell" in combined_lower or "dna" in combined_lower or "biology" in combined_lower:
            return True, "biology_diagram", "context_analysis"
        
        if "force" in combined_lower or "motion" in combined_lower or "circuit" in combined_lower:
            return True, "physics_diagram", "context_analysis"
        
        # No image needed based on RAG context
        return False, None, "no_image_needed"
    
    def generate_from_query(self, query: str, domain: str = "academic_tutoring") -> Optional[Dict[str, Any]]:
        """
        Generate an image based on a natural language query.
        
        Args:
            query: User's query requesting an image
            domain: Domain context (affects image style)
            
        Returns:
            Dictionary with image metadata or None
        """
        should_gen, img_type = self.should_generate_image(query)
        
        if not should_gen:
            return None
        
        try:
            if img_type == "function_graph":
                # Extract function expression from query
                expr = self._extract_function_expression(query)
                img_path = self.generate_function_graph(expr, title="Function Graph")
                
            elif img_type == "geometry":
                shape_params = self._extract_geometry_params(query)
                img_path = self.generate_geometry_diagram(shape_params["type"], shape_params["params"])
                
            elif img_type == "biology_diagram":
                img_path = self.generate_biology_diagram(query)
                
            elif img_type == "chemistry_diagram":
                img_path = self.generate_chemistry_diagram(query)
                
            elif img_type == "physics_diagram":
                img_path = self.generate_physics_diagram(query)
                
            elif img_type == "calculus_diagram":
                img_path = self.generate_calculus_diagram(query)
                
            else:
                # Default to function graph
                expr = self._extract_function_expression(query)
                img_path = self.generate_function_graph(expr)
            
            if img_path:
                return {
                    "image_url": f"/api/images/{img_path}",
                    "image_path": str(Path("images") / img_path),
                    "generated": True,
                    "type": img_type,
                    "filename": Path(img_path).name
                }
            
        except Exception as e:
            api_logger.error(f"Error generating image from query: {e}")
        
        return None
    
    def _extract_function_expression(self, query: str) -> str:
        """Extract function expression from query."""
        # Look for patterns like "y = x^2", "f(x) = sin(x)", etc.
        patterns = [
            r'[yf]\(?[x]\)?\s*=\s*([^,.\n]+)',
            r'graph\s+of\s+([^,.\n]+)',
            r'plot\s+([^,.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default
        return "x**2"
    
    def _extract_geometry_params(self, query: str) -> Dict[str, Any]:
        """Extract geometry parameters from query."""
        query_lower = query.lower()
        
        # Detect shape type
        if "triangle" in query_lower:
            shape_type = "triangle"
            # Extract angles
            angles = []
            angle_matches = re.findall(r'(\d+)\s*°', query)
            if angle_matches:
                angles = [int(a) for a in angle_matches[:3]]
                if len(angles) == 2:
                    remaining = 180 - sum(angles)
                    if remaining <= 0:
                        remaining = 180 - angles[0]
                    angles.append(remaining)
                elif len(angles) == 1:
                    remaining = 180 - angles[0]
                    second = remaining // 2
                    third = remaining - second
                    angles.extend([second, third])
            else:
                angles = [60, 60, 60]  # Default equilateral
            
            if len(angles) < 3:
                angles = (angles + [60, 60, 60])[:3]
            
            return {
                "type": shape_type,
                "params": {
                    "angles": angles,
                    "title": "Triangle"
                }
            }
        
        elif "circle" in query_lower:
            shape_type = "circle"
            # Extract radius
            radius_match = re.search(r'radius\s+(\d+)', query_lower)
            radius = int(radius_match.group(1)) if radius_match else 2
            
            return {
                "type": shape_type,
                "params": {
                    "radius": radius,
                    "title": f"Circle (r = {radius})"
                }
            }
        
        else:
            # Default rectangle
            return {
                "type": "rectangle",
                "params": {
                    "width": 4,
                    "height": 3,
                    "title": "Rectangle"
                }
            }
    
    def generate_biology_diagram(self, query: str) -> Optional[str]:
        """Generate a biology diagram (cell, DNA, molecules, etc.)."""
        try:
            query_lower = query.lower()
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect('equal')
            ax.axis('off')
            
            if "cell" in query_lower or "cellular" in query_lower:
                # Draw a simple cell diagram
                cell = Circle((0, 0), 3, fill=True, alpha=0.2, color='#3b82f6', edgecolor='#1e40af', linewidth=2)
                ax.add_patch(cell)
                nucleus = Circle((0, 0), 1.2, fill=True, alpha=0.4, color='#ef4444', edgecolor='#dc2626', linewidth=2)
                ax.add_patch(nucleus)
                ax.text(0, 0, 'Nucleus', ha='center', va='center', fontsize=12, fontweight='bold')
                mitochondria = mpatches.Ellipse((-1.5, -1.5), 0.8, 0.4, angle=45, 
                                              fill=True, alpha=0.3, color='#10b981', edgecolor='#059669', linewidth=1.5)
                ax.add_patch(mitochondria)
                ax.text(-1.5, -1.5, 'M', ha='center', va='center', fontsize=8)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                title = "Cell Structure"
            elif "dna" in query_lower:
                x1 = np.linspace(-2, 2, 100)
                y1 = 2 * np.sin(3 * x1)
                x2 = np.linspace(-2, 2, 100)
                y2 = -2 * np.sin(3 * x2)
                ax.plot(x1, y1, linewidth=2, color='#3b82f6')
                ax.plot(x2, y2, linewidth=2, color='#ef4444')
                for i in range(0, 100, 15):
                    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], 'k-', linewidth=1, alpha=0.5)
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-3, 3)
                title = "DNA Double Helix"
            else:
                atoms = [(-1, 0), (1, 0), (0, 1.5)]
                colors = ['#3b82f6', '#ef4444', '#10b981']
                labels = ['C', 'O', 'H']
                for i, ((x, y), color, label) in enumerate(zip(atoms, colors, labels)):
                    atom = Circle((x, y), 0.3, fill=True, color=color, edgecolor='black', linewidth=1.5)
                    ax.add_patch(atom)
                    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                ax.plot([-1, 1], [0, 0], 'k-', linewidth=2)
                ax.plot([0, 0], [0, 1.5], 'k-', linewidth=2)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1, 2)
                title = "Molecule Structure"
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            filename = f"generated_biology_{hash(query) % 10000}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            api_logger.info(f"Generated biology diagram: {filepath}")
            return str(filepath.relative_to(Path("images")))
        except Exception as e:
            api_logger.error(f"Error generating biology diagram: {e}")
            return None
    
    def generate_chemistry_diagram(self, query: str) -> Optional[str]:
        """Generate a chemistry diagram (molecules, reactions, etc.)."""
        try:
            query_lower = query.lower()
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Try to extract molecule name from query
            molecule_name = self._extract_molecule_name(query_lower)
            
            if "water" in query_lower or "h2o" in query_lower or "h₂o" in query_lower or molecule_name == "h2o":
                o_atom = Circle((0, 0), 0.4, fill=True, color='red', edgecolor='black', linewidth=2)
                ax.add_patch(o_atom)
                ax.text(0, 0, 'O', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
                h1 = Circle((-1.2, 0.7), 0.25, fill=True, color='lightblue', edgecolor='black', linewidth=1.5)
                h2 = Circle((1.2, 0.7), 0.25, fill=True, color='lightblue', edgecolor='black', linewidth=1.5)
                ax.add_patch(h1)
                ax.add_patch(h2)
                ax.text(-1.2, 0.7, 'H', ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(1.2, 0.7, 'H', ha='center', va='center', fontsize=10, fontweight='bold')
                ax.plot([-0.8, -1.05], [0.3, 0.6], 'k-', linewidth=3)
                ax.plot([0.8, 1.05], [0.3, 0.6], 'k-', linewidth=3)
                title = "Water Molecule (H₂O)"
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1, 1.5)
            elif "co2" in query_lower or "carbon dioxide" in query_lower:
                c_atom = Circle((0, 0), 0.35, fill=True, color='black', edgecolor='black', linewidth=2)
                ax.add_patch(c_atom)
                ax.text(0, 0, 'C', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                o1 = Circle((-1.3, 0), 0.3, fill=True, color='red', edgecolor='black', linewidth=2)
                o2 = Circle((1.3, 0), 0.3, fill=True, color='red', edgecolor='black', linewidth=2)
                ax.add_patch(o1)
                ax.add_patch(o2)
                ax.text(-1.3, 0, 'O', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
                ax.text(1.3, 0, 'O', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
                ax.plot([-0.9, -1.15], [0, 0], 'k-', linewidth=4)
                ax.plot([0.9, 1.15], [0, 0], 'k-', linewidth=4)
                title = "Carbon Dioxide (CO₂)"
                ax.set_xlim(-2, 2)
                ax.set_ylim(-0.8, 0.8)
            elif "nh3" in query_lower or "ammonia" in query_lower or molecule_name == "nh3":
                # Ammonia (NH3) - tetrahedral with 3 H atoms
                n_atom = Circle((0, 0), 0.35, fill=True, color='blue', edgecolor='black', linewidth=2)
                ax.add_patch(n_atom)
                ax.text(0, 0, 'N', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                h_positions = [(-1, 0.6), (1, 0.6), (0, -1)]
                for h_pos in h_positions:
                    h_atom = Circle(h_pos, 0.25, fill=True, color='lightblue', edgecolor='black', linewidth=1.5)
                    ax.add_patch(h_atom)
                    ax.text(h_pos[0], h_pos[1], 'H', ha='center', va='center', fontsize=9, fontweight='bold')
                    # Draw bond
                    ax.plot([h_pos[0]*0.7, h_pos[0]*0.9], [h_pos[1]*0.7, h_pos[1]*0.9], 'k-', linewidth=2)
                title = "Ammonia (NH₃)"
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1.5, 1.2)
            elif "ch4" in query_lower or "methane" in query_lower or molecule_name == "ch4":
                # Methane (CH4) - tetrahedral
                c_atom = Circle((0, 0), 0.35, fill=True, color='black', edgecolor='black', linewidth=2)
                ax.add_patch(c_atom)
                ax.text(0, 0, 'C', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                h_positions = [(0.9, 0.9), (-0.9, 0.9), (0.9, -0.9), (-0.9, -0.9)]
                for h_pos in h_positions:
                    h_atom = Circle(h_pos, 0.25, fill=True, color='lightblue', edgecolor='black', linewidth=1.5)
                    ax.add_patch(h_atom)
                    ax.text(h_pos[0], h_pos[1], 'H', ha='center', va='center', fontsize=9, fontweight='bold')
                    ax.plot([h_pos[0]*0.5, h_pos[0]*0.7], [h_pos[1]*0.5, h_pos[1]*0.7], 'k-', linewidth=2)
                title = "Methane (CH₄)"
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
            elif "nacl" in query_lower or "sodium chloride" in query_lower or "salt" in query_lower:
                # Sodium Chloride (NaCl) - ionic structure
                na_atom = Circle((-1, 0), 0.3, fill=True, color='yellow', edgecolor='black', linewidth=2)
                cl_atom = Circle((1, 0), 0.35, fill=True, color='green', edgecolor='black', linewidth=2)
                ax.add_patch(na_atom)
                ax.add_patch(cl_atom)
                ax.text(-1, 0, 'Na⁺', ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(1, 0, 'Cl⁻', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                ax.plot([-0.7, 0.7], [0, 0], 'k--', linewidth=2, alpha=0.5)
                title = "Sodium Chloride (NaCl)"
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1, 1)
            elif "reaction" in query_lower:
                ax.text(-2, 0, 'Reactants', ha='center', va='center', fontsize=12, fontweight='bold')
                ax.text(0, 0, '→', ha='center', va='center', fontsize=24, fontweight='bold')
                ax.text(2, 0, 'Products', ha='center', va='center', fontsize=12, fontweight='bold')
                ax.text(-2, -0.5, '2H₂ + O₂', ha='center', va='center', fontsize=11)
                ax.text(2, -0.5, '2H₂O', ha='center', va='center', fontsize=11)
                title = "Chemical Reaction"
                ax.set_xlim(-3, 3)
                ax.set_ylim(-1, 1)
            else:
                # Generic molecule - create a more informative structure
                # Instead of just "M", show a multi-atom molecule structure
                center = Circle((0, 0), 0.35, fill=True, color='#3b82f6', edgecolor='black', linewidth=2)
                ax.add_patch(center)
                ax.text(0, 0, 'C', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                
                # Add 3 connected atoms to show it's a molecule structure
                atom_positions = [(-1.2, 0.5), (1.2, 0.5), (0, -1)]
                for i, pos in enumerate(atom_positions):
                    atom = Circle(pos, 0.25, fill=True, color='lightblue', edgecolor='black', linewidth=1.5)
                    ax.add_patch(atom)
                    ax.text(pos[0], pos[1], 'H', ha='center', va='center', fontsize=9, fontweight='bold')
                    # Draw bond
                    ax.plot([pos[0]*0.7, pos[0]*0.9], [pos[1]*0.7, pos[1]*0.9], 'k-', linewidth=2)
                
                title = "Molecule Structure" if not molecule_name else f"{molecule_name.title()} Molecule"
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1.5, 1)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            filename = f"generated_chemistry_{hash(query) % 10000}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            api_logger.info(f"Generated chemistry diagram: {filepath} (molecule: {molecule_name or 'generic'})")
            return str(filepath.relative_to(Path("images")))
        except Exception as e:
            api_logger.error(f"Error generating chemistry diagram: {e}")
            return None
    
    def _extract_molecule_name(self, query_lower: str) -> Optional[str]:
        """Extract molecule name/formula from query."""
        # Common molecule patterns
        patterns = {
            r'\bh2o\b': 'h2o',
            r'\bco2\b': 'co2',
            r'\bnh3\b': 'nh3',
            r'\bch4\b': 'ch4',
            r'\bnacl\b': 'nacl',
            r'\bwater\b': 'h2o',
            r'\bcarbon\s+dioxide\b': 'co2',
            r'\bammonia\b': 'nh3',
            r'\bmethane\b': 'ch4',
            r'\bsodium\s+chloride\b': 'nacl',
            r'\bsalt\b': 'nacl',
        }
        
        for pattern, molecule in patterns.items():
            if re.search(pattern, query_lower):
                return molecule
        
        # Try to extract chemical formula pattern (e.g., "C6H12O6", "NaCl", "H2SO4")
        # Match uppercase letter followed by optional lowercase and optional numbers
        formula_match = re.search(r'\b([A-Z][a-z]?\d*)+([A-Z][a-z]?\d*)*\b', query_lower)
        if formula_match and len(formula_match.group(0)) > 1:
            formula = formula_match.group(0).lower()
            # Validate it's not just a single letter
            if len(formula) > 2 or (len(formula) == 2 and formula[1].isdigit()):
                return formula
        
        return None
    
    def generate_physics_diagram(self, query: str) -> Optional[str]:
        """Generate a physics diagram (forces, circuits, waves, etc.)."""
        try:
            query_lower = query.lower()
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if "wave" in query_lower:
                x = np.linspace(0, 4*np.pi, 1000)
                y = np.sin(x)
                ax.plot(x, y, linewidth=2, color='#2563eb')
                ax.fill_between(x, y, 0, alpha=0.3, color='#3b82f6')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Position', fontsize=12)
                ax.set_ylabel('Amplitude', fontsize=12)
                ax.set_title('Wave Motion', fontsize=14, fontweight='bold')
            elif "force" in query_lower or "free body" in query_lower:
                ax.axis('off')
                box = Rectangle((0.5, 0.5), 1, 1, fill=True, alpha=0.3, color='#3b82f6', edgecolor='#1e40af', linewidth=2)
                ax.add_patch(box)
                ax.text(1, 1, 'm', ha='center', va='center', fontsize=14, fontweight='bold')
                ax.arrow(1, 0.5, 0, -0.3, head_width=0.1, head_length=0.05, fc='red', ec='red', linewidth=2)
                ax.text(1.3, 0.35, 'mg', fontsize=10, color='red')
                ax.arrow(1, 1.5, 0, 0.2, head_width=0.1, head_length=0.05, fc='blue', ec='blue', linewidth=2)
                ax.text(1.3, 1.8, 'N', fontsize=10, color='blue')
                ax.set_xlim(0, 3)
                ax.set_ylim(0, 2.5)
                ax.set_title('Free Body Diagram', fontsize=14, fontweight='bold', pad=20)
            elif "circuit" in query_lower:
                ax.axis('off')
                battery = Rectangle((0.5, 0.3), 0.4, 0.4, fill=True, color='lightgray', edgecolor='black', linewidth=2)
                ax.add_patch(battery)
                ax.text(0.7, 0.5, '+', ha='center', va='center', fontsize=16, fontweight='bold')
                ax.plot([0.9, 3], [0.5, 0.5], 'k-', linewidth=3)
                resistor = Rectangle((2, 0.2), 0.5, 0.6, fill=True, color='brown', edgecolor='black', linewidth=2)
                ax.add_patch(resistor)
                ax.text(2.25, 0.5, 'R', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                ax.set_xlim(0, 3.5)
                ax.set_ylim(0, 1)
                ax.set_title('Simple Circuit', fontsize=14, fontweight='bold', pad=20)
            else:
                t = np.linspace(0, 10, 100)
                v = 5 * t
                ax.plot(t, v, linewidth=2, color='#2563eb')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Velocity (m/s)', fontsize=12)
                ax.set_title('Motion Diagram', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            filename = f"generated_physics_{hash(query) % 10000}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            api_logger.info(f"Generated physics diagram: {filepath}")
            return str(filepath.relative_to(Path("images")))
        except Exception as e:
            api_logger.error(f"Error generating physics diagram: {e}")
            return None
    
    def generate_calculus_diagram(self, query: str) -> Optional[str]:
        """Generate a calculus diagram (integral area, derivative slope, etc.)."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            query_lower = query.lower()
            
            if "integral" in query_lower:
                x = np.linspace(0, 5, 1000)
                y = x**2 / 2 + 1
                ax.plot(x, y, linewidth=2, color='#2563eb')
                ax.fill_between(x[:600], y[:600], 0, alpha=0.3, color='#3b82f6')
                ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5)
                ax.grid(True, alpha=0.3)
                ax.set_title('Definite Integral (Area under curve)', fontsize=14, fontweight='bold')
            elif "derivative" in query_lower:
                x = np.linspace(-2, 2, 1000)
                y = x**2
                ax.plot(x, y, linewidth=2, color='#2563eb', label='f(x) = x²')
                x_tangent = np.linspace(0, 2, 100)
                y_tangent = 2*(x_tangent - 1) + 1
                ax.plot(x_tangent, y_tangent, 'r--', linewidth=2, label='Tangent at x=1')
                ax.scatter([1], [1], color='red', s=100, zorder=5)
                ax.grid(True, alpha=0.3)
                ax.set_title('Derivative as Slope', fontsize=14, fontweight='bold')
                ax.legend()
            else:
                x = np.linspace(-3, 3, 1000)
                y = x**3 / 3
                ax.plot(x, y, linewidth=2, color='#2563eb', label='f(x)')
                ax.plot(x, x**2, linewidth=2, color='#ef4444', linestyle='--', label="f'(x)")
                ax.grid(True, alpha=0.3)
                ax.set_title('Function and Derivative', fontsize=14, fontweight='bold')
                ax.legend()
            
            plt.tight_layout()
            filename = f"generated_calculus_{hash(query) % 10000}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            api_logger.info(f"Generated calculus diagram: {filepath}")
            return str(filepath.relative_to(Path("images")))
        except Exception as e:
            api_logger.error(f"Error generating calculus diagram: {e}")
            return None


# Global instance
_image_generator = None

def get_image_generator() -> ImageGenerator:
    """Get or create global image generator instance."""
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator()
    return _image_generator

