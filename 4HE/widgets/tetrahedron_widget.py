import numpy as np
import pyqtgraph
import pyqtgraph.opengl as gl
from pyqtgraph.Transform3D import Transform3D
from pyqtgraph.opengl import GLTextItem
from PyQt5.QtGui import QVector3D, QFont
import math

class TetrahedronWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Adjusted camera settings for better visibility
        self.opts['distance'] = 3.0  # Closer view
        self.opts['elevation'] = 30  # Slightly higher angle
        self.opts['azimuth'] = 45    # Same azimuth
        self.setBackgroundColor((0.05, 0.05, 0.05, 1.0))
        
        self.volume_item = None
        self.core_scatter_item = None
        self.rim_scatter_item = None
        self._3d_labels_items = []

    def _generate_globular_cluster(self, radius, num_core, num_rim):
        """Generates points for a single globular cluster, centered at (0,0,0)."""
        # Generate core stars
        core_coords = np.random.normal(0, 1, (num_core, 3))
        core_coords *= (radius / 3.5)
        
        # Generate rim stars
        rim_coords = np.random.normal(0, 1, (num_rim, 3))
        rim_coords *= radius
        
        return core_coords, rim_coords

    def _get_font_size_for_tetra_label(self, simplicity):
        """
        Calculates font size based on simplicity. Lower simplicity means larger font.
        Example: "1:1:1:1" (simplicity 4) should be larger than "4:5:6:7" (simplicity 22).
        """
        # Define desired font size range in 3D units
        min_display_font_size = 0.8
        max_display_font_size = 4.0

        # Define expected simplicity range
        min_simplicity = 4 # For "1:1:1:1"
        max_simplicity = 50 # An estimated upper bound for typical odd-limits, can be adjusted

        # Clamp simplicity to the defined range
        clamped_simplicity = max(min_simplicity, min(max_simplicity, simplicity))

        # Linear inverse scaling:
        # As simplicity increases from min_simplicity to max_simplicity,
        # font_size decreases from max_display_font_size to min_display_font_size.
        
        # Calculate a normalized value (0 to 1) representing where simplicity falls in its range
        normalized_simplicity = (clamped_simplicity - min_simplicity) / (max_simplicity - min_simplicity)

        # Invert this normalized value (1 to 0) so lower simplicity maps to higher values
        inverted_normalized_simplicity = 1.0 - normalized_simplicity

        # Map the inverted normalized value to the desired font size range
        font_size = min_display_font_size + (inverted_normalized_simplicity * (max_display_font_size - min_display_font_size))
        
        return font_size

    def update_tetrahedron(self, volume_data, points_data, labels_data, show_volume, show_points, show_labels):
        # Clear previous items
        if self.volume_item: self.removeItem(self.volume_item); self.volume_item = None
        if self.core_scatter_item: self.removeItem(self.core_scatter_item); self.core_scatter_item = None
        if self.rim_scatter_item: self.removeItem(self.rim_scatter_item); self.rim_scatter_item = None
        
        # Clear existing labels
        for item in self._3d_labels_items:
            self.removeItem(item)
        self._3d_labels_items = []

        if not (show_volume or show_points or show_labels) or (volume_data is None and not labels_data):
            return
            
        x_grid, y_grid, z_grid, entropy = (None, None, None, None)
        if volume_data:
            x_grid, y_grid, z_grid, entropy = volume_data

        # --- Create Common Transformations ---
        # We need max_cents for scaling, which comes from the volume_data's grid.
        # If volume_data is not present, we need to derive max_cents from equave_ratio.
        # For now, assume volume_data is always generated if any visualization is active.
        # This might need refinement if labels are to be shown without volume data.
        # Get equave_ratio from gui.py
        equave_ratio_float = 2.0 # Default value
        if self.parent() and hasattr(self.parent(), 'equave_input'):
            try:
                equave_ratio_float = float(self.parent().equave_input.text())
            except ValueError:
                pass # Use default if conversion fails

        max_cents = 1200 * np.log2(equave_ratio_float) # Use actual equave_ratio
        if x_grid is not None:
            max_cents = x_grid.max() # Assuming x_grid represents c1, which goes up to max_cents

        res = 60 # Default resolution if volume_data is None
        if entropy is not None:
            res = entropy.shape[0]

        M = np.array([
            [1, 0.5, 0.5],
            [0, np.sqrt(3)/2, np.sqrt(3)/6],
            [0, 0, np.sqrt(6)/3]
        ])
        M4x4 = np.eye(4); M4x4[:3, :3] = M
        
        tetra_transform = Transform3D(M4x4)
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D(); translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])

        # --- Render Volume Data ---
        if show_volume and entropy is not None:
            # Create transform for volume (scales from grid space)
            scale_transform = Transform3D(); scale_transform.scale(1.0/res, 1.0/res, 1.0/res)
            volume_transform = translate_transform * tetra_transform * scale_transform

            min_e, max_e = np.nanmin(entropy), np.nanmax(entropy)
            norm_entropy = (entropy - min_e) / (max_e - min_e) if max_e > min_e else np.zeros_like(entropy)
            cmap = pyqtgraph.colormap.get('viridis')
            lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True); lut[:, 3] = np.linspace(0, 150, 256)
            safe_norm_entropy = norm_entropy.copy(); nan_mask = np.isnan(safe_norm_entropy); safe_norm_entropy[nan_mask] = 0
            colored_data = lut[(safe_norm_entropy * 255).astype(int)]; colored_data[nan_mask] = [0,0,0,0]
            
            self.volume_item = gl.GLVolumeItem(colored_data)
            self.volume_item.setTransform(volume_transform)
            self.addItem(self.volume_item)

        # --- Render Scatter Plot as Globular Clusters ---
        if show_points and points_data:
            # Create transform for scatter plot (assumes points are already in 0-1 space)
            scatter_transform = translate_transform * tetra_transform

            points = np.array(points_data)
            if len(points) == 0: return
                
            coords = points[:, :3]
            sums = points[:, 3]
            
            norm_coords = coords / max_cents

            min_s, max_s = sums.min(), sums.max()
            norm_s = 1.0 - ((sums - min_s) / (max_s - min_s + 1e-9))
            radii = (norm_s * 0.05) + 0.01

            all_core_points, all_rim_points = [], []
            for i in range(len(norm_coords)):
                center_point = norm_coords[i]
                radius = radii[i]
                # More important chords get more stars. Allocate more to the core.
                total_stars = int(norm_s[i] * 250) + 50
                num_core = int(total_stars * 0.6)
                num_rim = int(total_stars * 0.4)
                
                core_pts, rim_pts = self._generate_globular_cluster(radius, num_core, num_rim)
                all_core_points.extend(core_pts + center_point)
                all_rim_points.extend(rim_pts + center_point)

            if all_core_points:
                self.core_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_core_points), size=0.5, color=(1,1,1,0.9), pxMode=True)
                self.core_scatter_item.setTransform(scatter_transform)
                self.addItem(self.core_scatter_item)
            if all_rim_points:
                self.rim_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_rim_points), size=0.1, color=(1,1,1,0.7), pxMode=True)
                self.rim_scatter_item.setTransform(scatter_transform)
                self.addItem(self.rim_scatter_item)

        # --- Render Labels ---
        if show_labels and labels_data:
            labels_transform = translate_transform * tetra_transform
            
            for (c1, c2, c3), label_text, simplicity in labels_data:
                # Normalize cents coordinates to 0-1 range
                norm_c1 = c1 / max_cents
                norm_c2 = c2 / max_cents
                norm_c3 = c3 / max_cents

                # Create a temporary point in the 0-1 space
                temp_pos = np.array([norm_c1, norm_c2, norm_c3])

                # Convert numpy array to QVector3D for the map method
                temp_pos_qvector = QVector3D(float(temp_pos[0]), float(temp_pos[1]), float(temp_pos[2]))

                # Apply the transformation to get the final 3D position (QVector3D)
                transformed_pos_qvector = labels_transform.map(temp_pos_qvector)
                
                # Add a small Z-offset to bring labels forward
                transformed_pos_np = np.array([transformed_pos_qvector.x(), transformed_pos_qvector.y(), transformed_pos_qvector.z() + 0.2]) # Increased Z-offset

                font_size = self._get_font_size_for_tetra_label(simplicity)
                
                font = QFont()
                font.setPointSize(int(font_size * 100)) # Increased font size multiplier
                text_item = GLTextItem(pos=transformed_pos_np, text=label_text, color=(1.0, 1.0, 1.0, 1.0), font=font)
                self.addItem(text_item)

    def update_tetrahedron(self, volume_data, points_data, labels_data, show_volume, show_points, show_labels, font_size_multiplier):
        # Clear previous items
        if self.volume_item: self.removeItem(self.volume_item); self.volume_item = None
        if self.core_scatter_item: self.removeItem(self.core_scatter_item); self.core_scatter_item = None
        if self.rim_scatter_item: self.removeItem(self.rim_scatter_item); self.rim_scatter_item = None
        
        # Clear existing labels
        for item in self._3d_labels_items:
            self.removeItem(item)
        self._3d_labels_items = []

        # Check if any data is available for rendering
        # This ensures that if only points are requested, and points_data is None, it doesn't proceed
        # but if points are requested and points_data is available, it does.
        # Also, if nothing is requested, it should return.
        if not (show_volume or show_points or show_labels):
            return
        
        # If a mode is requested but its data is None, then don't try to render that mode.
        # This allows other modes to still render if their data is available.
        
        x_grid, y_grid, z_grid, entropy = (None, None, None, None)
        if show_volume and volume_data: # Only assign if show_volume is true and data exists
            x_grid, y_grid, z_grid, entropy = volume_data

        # --- Create Common Transformations ---
        # We need max_cents for scaling, which comes from the volume_data's grid.
        # If volume_data is not present, we need to derive max_cents from equave_ratio.
        # Get equave_ratio from gui.py
        equave_ratio_float = 2.0 # Default value
        if self.parent() and hasattr(self.parent(), 'equave_input'):
            try:
                equave_ratio_float = float(self.parent().equave_input.text())
            except ValueError:
                pass # Use default if conversion fails

        max_cents = 1200 * np.log2(equave_ratio_float) # Always use actual equave_ratio for max_cents

        res = 60 # Default resolution if volume_data is None
        if entropy is not None:
            res = entropy.shape[0]

        M = np.array([
            [1, 0.5, 0.5],
            [0, np.sqrt(3)/2, np.sqrt(3)/6],
            [0, 0, np.sqrt(6)/3]
        ])
        M4x4 = np.eye(4); M4x4[:3, :3] = M
        
        tetra_transform = Transform3D(M4x4)
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D(); translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])

        # --- Render Volume Data ---
        if show_volume and entropy is not None:
            # Create transform for volume (scales from grid space)
            scale_transform = Transform3D(); scale_transform.scale(1.0/res, 1.0/res, 1.0/res)
            volume_transform = translate_transform * tetra_transform * scale_transform

            min_e, max_e = np.nanmin(entropy), np.nanmax(entropy)
            norm_entropy = (entropy - min_e) / (max_e - min_e) if max_e > min_e else np.zeros_like(entropy)
            cmap = pyqtgraph.colormap.get('viridis')
            lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True); lut[:, 3] = np.linspace(0, 150, 256)
            safe_norm_entropy = norm_entropy.copy(); nan_mask = np.isnan(safe_norm_entropy); safe_norm_entropy[nan_mask] = 0
            colored_data = lut[(safe_norm_entropy * 255).astype(int)]; colored_data[nan_mask] = [0,0,0,0]
            
            self.volume_item = gl.GLVolumeItem(colored_data)
            self.volume_item.setTransform(volume_transform)
            self.addItem(self.volume_item)

        # --- Render Scatter Plot as Globular Clusters ---
        if show_points and points_data:
            # Create transform for scatter plot (assumes points are already in 0-1 space)
            scatter_transform = translate_transform * tetra_transform

            points = np.array(points_data)
            if len(points) == 0: # Don't return, just don't render scatter plot if no points
                pass
            else:
                coords = points[:, :3]
                sums = points[:, 3]
                
                norm_coords = coords / max_cents

                min_s, max_s = sums.min(), sums.max()
                norm_s = 1.0 - ((sums - min_s) / (max_s - min_s + 1e-9))
                radii = (norm_s * 0.05) + 0.01

                all_core_points, all_rim_points = [], []
                for i in range(len(norm_coords)):
                    center_point = norm_coords[i]
                    radius = radii[i]
                    # More important chords get more stars. Allocate more to the core.
                    total_stars = int(norm_s[i] * 250) + 50
                    num_core = int(total_stars * 0.6)
                    num_rim = int(total_stars * 0.4)
                    
                    core_pts, rim_pts = self._generate_globular_cluster(radius, num_core, num_rim)
                    all_core_points.extend(core_pts + center_point)
                    all_rim_points.extend(rim_pts + center_point)

                if all_core_points:
                    self.core_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_core_points), size=0.5, color=(1,1,1,0.9), pxMode=True)
                    self.core_scatter_item.setTransform(scatter_transform)
                    self.addItem(self.core_scatter_item)
                if all_rim_points:
                    self.rim_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_rim_points), size=0.1, color=(1,1,1,0.7), pxMode=True)
                    self.rim_scatter_item.setTransform(scatter_transform)
                    self.addItem(self.rim_scatter_item)

        # --- Render Labels ---
        if show_labels and labels_data:
            labels_transform = translate_transform * tetra_transform
            
            for (c1, c2, c3), label_text, simplicity in labels_data:
                # Normalize cents coordinates to 0-1 range
                norm_c1 = c1 / max_cents
                norm_c2 = c2 / max_cents
                norm_c3 = c3 / max_cents

                # Create a temporary point in the 0-1 space
                temp_pos = np.array([norm_c1, norm_c2, norm_c3])

                # Convert numpy array to QVector3D for the map method
                temp_pos_qvector = QVector3D(float(temp_pos[0]), float(temp_pos[1]), float(temp_pos[2]))

                # Apply the transformation to get the final 3D position (QVector3D)
                transformed_pos_qvector = labels_transform.map(temp_pos_qvector)
                
                # Add a small Z-offset to bring labels forward
                transformed_pos_np = np.array([transformed_pos_qvector.x(), transformed_pos_qvector.y(), transformed_pos_qvector.z() + 0.5]) # Increased Z-offset significantly

                font_size = self._get_font_size_for_tetra_label(simplicity)
                
                font = QFont()
                font.setPointSize(int(font_size * font_size_multiplier)) # Scale font_size to point size, increased multiplier
                text_item = GLTextItem(pos=transformed_pos_np, text=label_text, color=(255, 255, 255, 255), font=font) # Corrected color range
                self.addItem(text_item)
                self._3d_labels_items.append(text_item)

                # Remove temporary red dots
                # dot_item = gl.GLScatterPlotItem(pos=np.array([transformed_pos_np[:3]]), size=0.1, color=(1.0, 0.0, 0.0, 1.0), pxMode=False)
                # self.addItem(dot_item)
                # self._3d_labels_items.append(dot_item) # Add to labels_items so it gets cleared

                # Remove temporary red dots
                # dot_item = gl.GLScatterPlotItem(pos=np.array([transformed_pos_np[:3]]), size=0.1, color=(1.0, 0.0, 0.0, 1.0), pxMode=False)
                # self.addItem(dot_item)
                # self._3d_labels_items.append(dot_item) # Add to labels_items so it gets cleared