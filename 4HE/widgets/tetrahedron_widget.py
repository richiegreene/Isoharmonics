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
        self.opts['distance'] = 3.0
        self.opts['elevation'] = 30
        self.opts['azimuth'] = 45
        self.setBackgroundColor((0.05, 0.05, 0.05, 1.0))
        
        self.volume_item = None
        self.scatter_item = None
        self._3d_labels_items = []

    def update_tetrahedron(self, volume_data, points_data, labels_data, show_volume, show_points, show_labels):
        if self.volume_item: self.removeItem(self.volume_item); self.volume_item = None
        if self.scatter_item: self.removeItem(self.scatter_item); self.scatter_item = None
        
        for item in self._3d_labels_items:
            self.removeItem(item)
        self._3d_labels_items = []

        if not (show_volume or show_points or show_labels):
            return
        
        equave_ratio_float = 2.0
        if self.parent() and hasattr(self.parent(), 'equave_input'):
            try:
                equave_ratio_float = float(self.parent().equave_input.text())
            except ValueError:
                pass

        max_cents = 1200 * np.log2(equave_ratio_float)

        M = np.array([[1, 0.5, 0.5], [0, np.sqrt(3)/2, np.sqrt(3)/6], [0, 0, np.sqrt(6)/3]])
        M4x4 = np.eye(4); M4x4[:3, :3] = M
        tetra_transform = Transform3D(M4x4)
        center_of_mass = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/12])
        translate_transform = Transform3D(); translate_transform.translate(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])

        if show_volume and volume_data:
            x_grid, y_grid, z_grid, entropy = volume_data
            res = entropy.shape[0]
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

        if show_points and points_data:
            scatter_transform = translate_transform * tetra_transform
            points = np.array(points_data)
            if len(points) > 0:
                coords = points[:, :3]
                complexities = points[:, 3]
                norm_coords = coords / max_cents
                
                min_c, max_c = complexities.min(), complexities.max()
                if max_c > min_c:
                    norm_c = 1.0 - ((complexities - min_c) / (max_c - min_c))
                else:
                    norm_c = np.ones_like(complexities)
                
                sizes = (norm_c * 7) + 1 # Sizes from 1 to 8

                self.scatter_item = gl.GLScatterPlotItem(pos=norm_coords, size=sizes, color=(1,1,1,0.8), pxMode=False)
                self.scatter_item.setTransform(scatter_transform)
                self.addItem(self.scatter_item)

        if show_labels and labels_data:
            labels_transform = translate_transform * tetra_transform
            complexities = np.array([item[2] for item in labels_data])
            min_c, max_c = complexities.min(), complexities.max()

            for (c1, c2, c3), label_text, complexity in labels_data:
                norm_c1 = c1 / max_cents
                norm_c2 = c2 / max_cents
                norm_c3 = c3 / max_cents

                temp_pos = np.array([norm_c1, norm_c2, norm_c3])
                temp_pos_qvector = QVector3D(float(temp_pos[0]), float(temp_pos[1]), float(temp_pos[2]))
                transformed_pos_qvector = labels_transform.map(temp_pos_qvector)
                transformed_pos_np = np.array([transformed_pos_qvector.x(), transformed_pos_qvector.y(), transformed_pos_qvector.z() + 0.5])

                if max_c > min_c:
                    norm_c = 1.0 - ((complexity - min_c) / (max_c - min_c))
                else:
                    norm_c = 1.0
                
                font_size = int((norm_c * 8) + 8) # Font sizes from 8 to 16

                font = QFont()
                font.setPointSize(font_size)
                text_item = GLTextItem(pos=transformed_pos_np, text=label_text, color=(255, 255, 255, 255), font=font)
                self.addItem(text_item)
                self._3d_labels_items.append(text_item)