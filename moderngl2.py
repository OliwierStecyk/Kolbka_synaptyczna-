import time
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44
import imageio.v3 as iio 

class ModernGLRenderer:
    def __init__(self, xxx, yyy, zzz, 
                 xfliml, xflimh, yfliml, yflimh, zfliml, zflimh,
                 size=(1920, 1080), point_size=3.0):
        
        # Inicjalizacja kontekstu OpenGL (tylko raz)
        self.ctx = moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self.size = size
        self.point_size = point_size
        
        # Przygotowanie buforów wierzchołków (tylko raz)
        self.vertices = np.column_stack([xxx, yyy, zzz]).astype('f4')
        self.vbo_verts = self.ctx.buffer(self.vertices.tobytes())
        
        # Kompilacja programu (tylko raz)
        self.prog = self.ctx.program(
            vertex_shader=self._get_vertex_shader(),
            fragment_shader=self._get_fragment_shader()
        )
        self.grid_prog = self.ctx.program(vertex_shader=self._get_grid_vertex_shader(), 
                                     fragment_shader=self._get_grid_fragment_shader())
        
        # Inicjalizacja VAO (bez danych o kolorach)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_verts, '3f4', 'in_position')
            ]
        )
        self.grid_vao = self.create_single_grid_vao(
            self.ctx, self.grid_prog,
            x_range=(xfliml, xflimh),
            y_range=(yfliml, yflimh),
            z_fixed=zfliml,  # jedna płaska kratka na dolnej granicy osi Z
            step=(xflimh - xfliml) / 10.0
        )
        
        # Przygotowanie FBO (tylko raz)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(size, 4)],
            depth_attachment=self.ctx.depth_texture(size)
        )
        
        # Ustawienia kamery (tylko raz)
        self._setup_camera(xfliml, xflimh, yfliml, yflimh, zfliml, zflimh)
    
    def _get_vertex_shader(self):
        return f"""
        #version 330
        uniform mat4 mvp;
        in vec3 in_position;
        in float in_color;
        out float v_color;
        void main() {{
            gl_Position = mvp * vec4(in_position, 1.0);
            v_color = in_color;
            gl_PointSize = {self.point_size};
        }}
        """
    def _get_fragment_shader(self):
        return """
        #version 330
        uniform float vmin;
        uniform float vmax;
        in float v_color;
        out vec4 fragColor;
        
        vec3 hot_colormap(float t) {
            return vec3(
                min(1.0, max(0.0, t * 4.0 - 1.5)), // R
                min(1.0, max(0.0, t * 2.0 - 0.5)), // G
                min(1.0, max(0.0, t * 2.0))       // B
            );
        }
        
        void main() {
            vec3 color = hot_colormap(v_color);
            fragColor = vec4(color, 1.0);
        }
        """
    def _get_grid_vertex_shader(self):
        return """
        #version 330
        uniform mat4 mvp;
        in vec3 in_position;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
        }
        """
    def _get_grid_fragment_shader(self):
        return """
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // czarna siatka
        }
        """
    
    def _setup_camera(self, xfliml, xflimh, yfliml, yflimh, zfliml, zflimh):
        center = np.array([np.mean([xfliml, xflimh]),
                          np.mean([yfliml, yflimh]), 
                          np.mean([zfliml, zflimh])])
        radius = max(xflimh-xfliml, yflimh-yfliml, zflimh-zfliml) * 1.5
        proj = Matrix44.perspective_projection(fovy=30, aspect=self.size[0]/self.size[1], near=0.1, far=radius*10)
        #fovy odpowiada za oddalenie
        view = Matrix44.look_at(
            eye=(xfliml-radius, yfliml-radius, zflimh+radius),
            #(center[0], center[1], center[2] + radius),
            target=center,
            up=(1,0,0)
        )
        self.mvp = proj * view
    
    def render(self, vvv, output_path, vmin=200, vmax=450):
        # Normalizacja kolorów
        normalized_colors = ((vvv - vmin) / (vmax - vmin)).astype('f4')
        
        # Aktualizacja bufora kolorów
        vbo_colors = self.ctx.buffer(normalized_colors.tobytes())
        
        # Aktualizacja VAO z nowymi kolorami
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_verts, '3f4', 'in_position'),
                (vbo_colors, '1f4', 'in_color')
            ]
        )
        
        # Renderowanie
        self.fbo.use()
        self.fbo.clear(1.0, 1.0, 1.0, 1.0)
        self.prog['mvp'].write(self.mvp.astype('f4').tobytes())
        self.grid_prog['mvp'].write(self.mvp.astype('f4').tobytes())
        self.vao.render(moderngl.POINTS)
        self.grid_vao.render(moderngl.LINES)

        # Pobranie obrazu
        img_data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.size, img_data)
        img = self.add_colorbar(img, vmin, vmax)
        
        # Zapis
        iio.imwrite(output_path, image=img, compression=6)
        
        # Zwolnienie zasobów
        vbo_colors.release()
    
    def release(self):
        self.vao.release()
        self.vbo_verts.release()
        self.prog.release()
        self.fbo.release()
        self.ctx.release()

    def add_colorbar(self, img, vmin, vmax, width=100):
        """Dodaje colorbar do obrazu używając PIL"""
        # Utwórz gradient
        height = img.height
        gradient = np.linspace(1, 0, height).reshape(height, 1)
        gradient = np.tile(gradient, (1, width))
        
        # Mapowanie kolorów (hot)
        hot_colors = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):                                     # do naprawienia (usunąc fora jakoś)
            val = 1 - i/height
            r = min(255, max(0, int(val * 4 * 255 - 1.5 * 255)))
            g = min(255, max(0, int(val * 2 * 255 - 0.5 * 255)))
            b = min(255, max(0, int(val * 2 * 255)))
            hot_colors[i, :] = [r, g, b]
        
        # Stwórz finalny obraz
        colorbar_img = Image.fromarray(hot_colors, 'RGB')
        final_img = Image.new('RGB', (img.width + width, height))
        final_img.paste(img, (0, 0))
        final_img.paste(colorbar_img, (img.width, 0))
        
        # Dodaj etykiety
        draw = ImageDraw.Draw(final_img)

        # Ustaw większą czcionkę
        font = ImageFont.truetype("arial.ttf", 28)

        draw.text((img.width - 90, 10), str(vmax), fill=(0,0,0), font=font)
        draw.text((img.width - 90, height-40), str(vmin), fill=(0,0,0), font=font)
        
        return final_img
    
    def create_single_grid_vao(self, ctx, grid_prog, x_range, y_range, z_fixed, step=1.0):
        lines = []

        # Linie równoległe do osi X na płaszczyźnie XY, dla stałego z = z_fixed
        for y in np.arange(y_range[0], y_range[1] + step, step):
            lines.append([x_range[0], y, z_fixed])
            lines.append([x_range[1], y, z_fixed])

        # Linie równoległe do osi Y na płaszczyźnie XY, dla stałego z = z_fixed
        for x in np.arange(x_range[0], x_range[1] + step, step):
            lines.append([x, y_range[0], z_fixed])
            lines.append([x, y_range[1], z_fixed])

        grid_vertices = np.array(lines, dtype='f4')
        vbo = ctx.buffer(grid_vertices.tobytes())
        vao = ctx.simple_vertex_array(grid_prog, vbo, 'in_position')
        return vao


# Przykład użycia:
if __name__ == "__main__":
    # Generowanie danych testowych
    n = 3102
    xxx = np.random.rand(n) * 10
    yyy = np.random.rand(n) * 10
    zzz = np.random.rand(n) * 10
    
    # Inicjalizacja renderera (tylko raz)
    renderer = ModernGLRenderer(xxx, yyy, zzz, 0,10,0,10,0,10)
    
    try:
        # Generowanie wielu wykresów
        for i in range(5):
            vvv = np.random.rand(n) * 25 + 340  # Nowe wartości kolorów
            output_path = f"./plots/output_{i}.png"
            
            t0 = time.time()
            renderer.render(vvv, output_path)
            print(f"Render {i} czas: {time.time()-t0:.4f}s path: {output_path}")
    finally:
        renderer.release()