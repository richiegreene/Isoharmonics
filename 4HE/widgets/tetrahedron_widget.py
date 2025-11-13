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

    def update_tetrahedron(self, data, ji_data):
        if self.volume_item:
            self.removeItem(self.volume_item)

        if data is None:
            return
            
        x_grid, y_grid, z_grid, entropy = data
        
        if entropy is None:
            return

        # Normalize entropy and create a color map (lookup table)
        min_e, max_e = np.nanmin(entropy), np.nanmax(entropy)
        if max_e - min_e > 0:
            norm_entropy = (entropy - min_e) / (max_e - min_e)
        else:
            norm_entropy = np.zeros_like(entropy)
        
        # Use a colormap where low values are transparent
        cmap = pyqtgraph.colormap.get('viridis')
        lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True)
        lut[:, 3] = np.linspace(0, 150, 256) # Set alpha channel
        
        # Create a copy of the normalized entropy and replace NaNs with a safe value (0)
        # before converting to integers for indexing the lookup table.
        safe_norm_entropy = norm_entropy.copy()
        nan_mask = np.isnan(safe_norm_entropy)
        safe_norm_entropy[nan_mask] = 0
        
        # Apply the colormap using the safe, non-NaN values
        colored_data = lut[(safe_norm_entropy * 255).astype(int)]
        
        # Now, make the original NaN locations fully transparent
        colored_data[nan_mask] = [0, 0, 0, 0]
        
        # Create the volume item
        # GLVolumeItem expects data in (z,y,x) order, but our data is already in that logical order
        self.volume_item = gl.GLVolumeItem(colored_data)
        
        # --- Transformation to shape the cube into a tetrahedron ---
        res = entropy.shape[0]
        
        # This matrix maps the cartesian axes of the data cube to the basis vectors
        # of a regular tetrahedron.
        M = np.array([
            [1, 0.5,              0.5],
            [0, np.sqrt(3)/2,     np.sqrt(3)/6],
            [0, 0,                np.sqrt(6)/3]
        ])

        # pyqtgraph's Transform3D requires a 4x4 matrix.
        M4x4 = np.eye(4)
        M4x4[:3, :3] = M

        # 1. Scale the volume from (res,res,res) to a unit cube (1,1,1)
        scale_transform = Transform3D()
        scale_transform.scale(1.0/res, 1.0/res, 1.0/res)

        # 2. Apply the tetrahedral shear transformation
        tetra_transform = Transform3D(M4x4)

        # 3. Center the resulting model at the origin
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D()
        translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])

        # Combine transforms. They are applied right-to-left: scale -> shear -> translate
        final_transform = translate_transform * tetra_transform * scale_transform

        self.volume_item.setTransform(final_transform)
        self.addItem(self.volume_item)