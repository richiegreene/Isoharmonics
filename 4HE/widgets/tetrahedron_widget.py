import numpy as np
import pyqtgraph
import pyqtgraph.opengl as gl
from pyqtgraph.Transform3D import Transform3D

class TetrahedronWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['distance'] = 4.5
        self.opts['elevation'] = 20
        self.opts['azimuth'] = 45
        self.setBackgroundColor((0.05, 0.05, 0.05, 1.0))
        
        self.volume_item = None
        self.core_scatter_item = None
        self.rim_scatter_item = None

    def _generate_globular_cluster(self, radius, num_core, num_rim):
        """Generates points for a single globular cluster, centered at (0,0,0)."""
        # Generate core stars
        core_coords = np.random.normal(0, 1, (num_core, 3))
        core_coords *= (radius / 2.5)
        
        # Generate rim stars
        rim_coords = np.random.normal(0, 1, (num_rim, 3))
        rim_coords *= radius
        
        return core_coords, rim_coords

    def update_tetrahedron(self, volume_data, points_data, show_volume, show_points):
        # Clear previous items
        if self.volume_item: self.removeItem(self.volume_item); self.volume_item = None
        if self.core_scatter_item: self.removeItem(self.core_scatter_item); self.core_scatter_item = None
        if self.rim_scatter_item: self.removeItem(self.rim_scatter_item); self.rim_scatter_item = None

        if not (show_volume or show_points) or volume_data is None:
            return
            
        x_grid, y_grid, z_grid, entropy = volume_data
        if entropy is None: return

        # --- Transformation (calculated from volume data) ---
        res = entropy.shape[0]
        M = np.array([
            [1, 0.5, 0.5],
            [0, np.sqrt(3)/2, np.sqrt(3)/6],
            [0, 0, np.sqrt(6)/3]
        ])
        M4x4 = np.eye(4); M4x4[:3, :3] = M
        
        scale_transform = Transform3D(); scale_transform.scale(1.0/res, 1.0/res, 1.0/res)
        tetra_transform = Transform3D(M4x4)
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D(); translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])
        final_transform = translate_transform * tetra_transform * scale_transform

        # --- Render Volume Data ---
        if show_volume:
            min_e, max_e = np.nanmin(entropy), np.nanmax(entropy)
            norm_entropy = (entropy - min_e) / (max_e - min_e) if max_e > min_e else np.zeros_like(entropy)
            cmap = pyqtgraph.colormap.get('viridis')
            lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True); lut[:, 3] = np.linspace(0, 150, 256)
            safe_norm_entropy = norm_entropy.copy(); nan_mask = np.isnan(safe_norm_entropy); safe_norm_entropy[nan_mask] = 0
            colored_data = lut[(safe_norm_entropy * 255).astype(int)]; colored_data[nan_mask] = [0,0,0,0]
            
            self.volume_item = gl.GLVolumeItem(colored_data)
            self.volume_item.setTransform(final_transform)
            self.addItem(self.volume_item)

        # --- Render Scatter Plot as Globular Clusters ---
        if show_points and points_data:
            points = np.array(points_data)
            if len(points) == 0: return
                
            coords = points[:, :3]
            sums = points[:, 3]
            
            max_cents = x_grid.max()
            norm_coords = coords / max_cents

            min_s, max_s = sums.min(), sums.max()
            norm_s = 1.0 - ((sums - min_s) / (max_s - min_s + 1e-9))
            radii = (norm_s * 0.05) + 0.01

            all_core_points = []
            all_rim_points = []

            for i in range(len(norm_coords)):
                center_point = norm_coords[i]
                radius = radii[i]
                num_stars = int(norm_s[i] * 200) + 50
                
                core_pts, rim_pts = self._generate_globular_cluster(radius, int(num_stars/4), num_stars)
                all_core_points.extend(core_pts + center_point)
                all_rim_points.extend(rim_pts + center_point)

            if all_core_points:
                self.core_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_core_points), size=0.5, color=(1,1,1,0.9), pxMode=True)
                self.core_scatter_item.setTransform(final_transform)
                self.addItem(self.core_scatter_item)
            if all_rim_points:
                self.rim_scatter_item = gl.GLScatterPlotItem(pos=np.array(all_rim_points), size=0.1, color=(1,1,1,0.7), pxMode=True)
                self.rim_scatter_item.setTransform(final_transform)
                self.addItem(self.rim_scatter_item)