import numpy as np
import pyqtgraph
import pyqtgraph.opengl as gl
from pyqtgraph.Transform3D import Transform3D

class TetrahedronWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['distance'] = 3.5
        self.opts['elevation'] = 20
        self.opts['azimuth'] = 45
        self.setBackgroundColor((0.1, 0.1, 0.1, 1.0))
        
        self.volume_item = None
        self.points_item = None

    def update_tetrahedron(self, volume_data, points_data, show_volume, show_points):
        # Clear previous items
        if self.volume_item:
            self.removeItem(self.volume_item)
            self.volume_item = None
        if self.points_item:
            self.removeItem(self.points_item)
            self.points_item = None

        if not (show_volume or show_points):
            return

        if volume_data is None:
            return
            
        x_grid, y_grid, z_grid, entropy = volume_data
        
        if entropy is None:
            return

        # --- Transformation (calculated from volume data) ---
        res = entropy.shape[0]
        M = np.array([
            [1, 0.5,              0.5],
            [0, np.sqrt(3)/2,     np.sqrt(3)/6],
            [0, 0,                np.sqrt(6)/3]
        ])
        M4x4 = np.eye(4)
        M4x4[:3, :3] = M
        
        scale_transform = Transform3D()
        scale_transform.scale(1.0/res, 1.0/res, 1.0/res)
        tetra_transform = Transform3D(M4x4)
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D()
        translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])
        final_transform = translate_transform * tetra_transform * scale_transform

        # --- Render Volume Data ---
        if show_volume:
            min_e, max_e = np.nanmin(entropy), np.nanmax(entropy)
            if max_e - min_e > 0:
                norm_entropy = (entropy - min_e) / (max_e - min_e)
            else:
                norm_entropy = np.zeros_like(entropy)
            
            cmap = pyqtgraph.colormap.get('viridis')
            lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True)
            lut[:, 3] = np.linspace(0, 150, 256)
            
            safe_norm_entropy = norm_entropy.copy()
            nan_mask = np.isnan(safe_norm_entropy)
            safe_norm_entropy[nan_mask] = 0
            
            colored_data = lut[(safe_norm_entropy * 255).astype(int)]
            colored_data[nan_mask] = [0, 0, 0, 0]
            
            self.volume_item = gl.GLVolumeItem(colored_data)
            self.volume_item.setTransform(final_transform)
            self.addItem(self.volume_item)

        # --- Render Data Points ---
        if show_points and points_data:
            points = np.array(points_data)
            if len(points) == 0:
                return
                
            coords = points[:, :3]
            sums = points[:, 3]
            
            # Apply the same transformation to the points
            # Note: The points are in cents, so we need to scale them by max_cents first
            max_cents = x_grid.max()
            scaled_coords = coords / max_cents
            
            # Transform points from a unit cube space to the tetrahedron shape
            # We need to build a 4D homogenous coordinate to apply the 4x4 matrix
            homogenous_coords = np.hstack([scaled_coords, np.ones((len(scaled_coords), 1))])
            
            # Apply scale -> shear -> translate
            transformed_homogenous = (final_transform.matrix() @ homogenous_coords.T).T
            transformed_coords = transformed_homogenous[:, :3]

            # Calculate size based on inverse sum
            min_s, max_s = sums.min(), sums.max()
            # Normalize sum: smaller sum -> larger value
            norm_s = 1.0 - ((sums - min_s) / (max_s - min_s + 1e-9))
            sizes = (norm_s * 10) + 2 # Map to a reasonable size range [2, 12]

            colors = np.array([[1.0, 1.0, 1.0, 0.4]] * len(transformed_coords)) # Translucent white

            self.points_item = gl.GLScatterPlotItem(pos=transformed_coords, size=sizes, color=colors, pxMode=False)
            self.addItem(self.points_item)
