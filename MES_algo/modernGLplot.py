import time
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44
import imageio.v3 as iio


class ModernGLRenderer:
    def __init__(self, xxx, yyy, zzz,
                 xfliml, xflimh, yfliml, yflimh, zfliml, zflimh, vmin=300, vmax=450,
                 size=(1280, 720), point_size=5.0, grid_steps = 10):
        ## point_size dużo ładniejsze na 5 // wczesniej 3

        ## narazie zrobie tak ale pozniej mozna to dodac do init zeby bylo latwiej do modyfikacji
        self.grid_steps = grid_steps
        self.vmin = vmin
        self.vmax = vmax
        
        ## ======================= nowa rzecz
        self.xfliml, self.xflimh = xfliml, xflimh
        self.yfliml, self.yflimh = yfliml, yflimh
        self.zfliml, self.zflimh = zfliml, zflimh
        ## ======================= nowa rzecz


        self.size = size
        self.point_size = point_size

        # Inicjalizacja kontekstu OpenGL (tylko raz)
        self.ctx = moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)

        # Przygotowanie buforów wierzchołków (tylko raz)
        self.vertices = np.column_stack([xxx, yyy, zzz]).astype('f4')
        self.vbo_verts = self.ctx.buffer(self.vertices.tobytes())


        self.prog = self.ctx.program(
            vertex_shader=self._get_vertex_shader(),
            fragment_shader=self._get_fragment_shader()
        )
        self.grid_prog = self.ctx.program(
            vertex_shader=self._get_grid_vertex_shader(),
            fragment_shader=self._get_grid_fragment_shader()
        )

        ### nowe
        self.grid_vaos = [
            self.create_grid_vao('xy', self.zfliml),
            self.create_grid_vao('yz', self.xflimh),
            self.create_grid_vao('xz', self.yflimh)
        ]
        self._setup_camera()
        self.colorbar = self.create_colorbar()
        ### nowe

        # # Inicjalizacja VAO (bez danych o kolorach)
        # self.vao = self.ctx.vertex_array(
        #     self.prog,
        #     [
        #         (self.vbo_verts, '3f4', 'in_position')
        #     ]
        # )
        # self.grid_vao = self.create_single_grid_vao(
        #     self.ctx, self.grid_prog,
        #     x_range=(xfliml, xflimh),
        #     y_range=(yfliml, yflimh),
        #     z_fixed=zfliml,  # jedna płaska kratka na dolnej granicy osi Z
        #     step=(xflimh - xfliml) / 10.0
        # ) stare

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(size, 4)],
            depth_attachment=self.ctx.depth_renderbuffer(size)
        )

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

    def _get_fragment_shader(self): #### tutaj zmieniamy kolorystyke
        return """
        #version 330
        in float v_color;
        out vec4 fragColor;

        vec3 warm_colormap(float t) {
            // Nowe mapowanie kolorów:
            // - Czerwony szybko osiąga maksimum
            // - Zieleń stopniowo wzrasta
            // - Brak niebieskiego w niższych wartościach
            return vec3(
                smoothstep(0.0, 1.0, t * 3.0 - 0.5),
                smoothstep(0.0, 1.0, t * 1.5 - 0.25),
                smoothstep(0.0, 1.0, t * 0.75)
            );
        }

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = dot(coord, coord);
            if (dist > 0.25) discard;
            vec3 color = warm_colormap(v_color);
            fragColor = vec4(color, 1.0);
        }
        """

    # def _get_fragment_shader(self):
    #     return """
    #        #version 330
    #        uniform float vmin;
    #        uniform float vmax;
    #        in float v_color;
    #        out vec4 fragColor;
    #
    #        vec3 hot_colormap(float t) {
    #            return vec3(
    #                min(1.0, max(0.0, t * 4.0 - 1.5)), // R
    #                min(1.0, max(0.0, t * 2.0 - 0.5)), // G
    #                min(1.0, max(0.0, t * 2.0))       // B
    #            );
    #        }
    #
    #        void main() {
    #            vec3 color = hot_colormap(v_color);
    #            fragColor = vec4(color, 1.0);
    #        }
    #        """ stare

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
            fragColor = vec4(0.2, 0.2, 0.2, 1.0);
        }
        """

    def _setup_camera(self):
        center = [
            (self.xfliml + self.xflimh) / 2,
            (self.yfliml + self.yflimh) / 2,
            (self.zfliml + self.zflimh) / 2
        ]
        radius = max(
            self.xflimh - self.xfliml,
            self.yflimh - self.yfliml,
            self.zflimh - self.zfliml
        ) * 1.5
        proj = Matrix44.perspective_projection(fovy=30.0, aspect=self.size[0] / self.size[1], near=0.1, far=radius * 10)
        view = Matrix44.look_at(
            eye=(self.xfliml - radius, self.yfliml - radius, self.zflimh + radius),
            target=center,
            up=(1.0, 0.0, 0.0)
        )
        self.mvp = proj * view

    # def _setup_camera(self, xfliml, xflimh, yfliml, yflimh, zfliml, zflimh):
    #     center = np.array([np.mean([xfliml, xflimh]),
    #                        np.mean([yfliml, yflimh]),
    #                        np.mean([zfliml, zflimh])])
    #     radius = max(xflimh - xfliml, yflimh - yfliml, zflimh - zfliml) * 1.5
    #     proj = Matrix44.perspective_projection(fovy=30, aspect=self.size[0] / self.size[1], near=0.1, far=radius * 10)
    #     # fovy odpowiada za oddalenie
    #     view = Matrix44.look_at(
    #         eye=(xfliml - radius, yfliml - radius, zflimh + radius),
    #         # (center[0], center[1], center[2] + radius),
    #         target=center,
    #         up=(1, 0, 0)
    #     )
    #     self.mvp = proj * view
    #   stare


    def render(self, vvv, output_path):
        #t = time.time()

        colors = ((vvv - self.vmin)/(self.vmax - self.vmin)).astype('f4')
        vbo_colors = self.ctx.buffer(colors.tobytes())

        vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_verts, '3f4', 'in_position'),
                (vbo_colors, '1f4', 'in_color')
            ]
        )
        
        self.fbo.use()
        self.fbo.clear(1.0, 1.0, 1.0, 1.0)

        self.prog['mvp'].write(self.mvp.astype('f4').tobytes())
        self.grid_prog['mvp'].write(self.mvp.astype('f4').tobytes())

        for grid_vao in self.grid_vaos:
            grid_vao.render(moderngl.LINES)

        vao.render(moderngl.POINTS)

        img = Image.frombytes('RGB', self.size, self.fbo.read(components=3))

        #print(f"Image created in {time.time() - t:.4f}s")

        img = self.add_colorbar(img, self.colorbar)
        iio.imwrite(output_path, image=img, compression=6)
        vbo_colors.release()



    # def render(self, vvv, output_path, vmin=200, vmax=450):
    #     # Normalizacja kolorów
    #     normalized_colors = ((vvv - vmin) / (vmax - vmin)).astype('f4')
    #
    #     # Aktualizacja bufora kolorów
    #     vbo_colors = self.ctx.buffer(normalized_colors.tobytes())
    #
    #     # Aktualizacja VAO z nowymi kolorami
    #     self.vao = self.ctx.vertex_array(
    #         self.prog,
    #         [
    #             (self.vbo_verts, '3f4', 'in_position'),
    #             (vbo_colors, '1f4', 'in_color')
    #         ]
    #     )
    #
    #     # Renderowanie
    #     self.fbo.use()
    #     self.fbo.clear(1.0, 1.0, 1.0, 1.0)
    #     self.prog['mvp'].write(self.mvp.astype('f4').tobytes())
    #     self.grid_prog['mvp'].write(self.mvp.astype('f4').tobytes())
    #     self.vao.render(moderngl.POINTS)
    #     self.grid_vao.render(moderngl.LINES)
    #
    #     # Pobranie obrazu
    #     img_data = self.fbo.read(components=3)
    #     img = Image.frombytes('RGB', self.size, img_data)
    #     img = self.add_colorbar(img, vmin, vmax)
    #
    #     # Zapis
    #     iio.imwrite(output_path, image=img, compression=6)
    #
    #     # Zwolnienie zasobów
    #     vbo_colors.release()
    def release(self):
        if self.grid_vaos:
            self.grid_vaos.release()
        if self.vbo_verts:
            self.vbo_verts.release()
        if self.prog:
            self.prog.release()
        if self.fbo:
            self.fbo.release()
        if self.ctx:
            self.ctx.release()


    def create_colorbar(self, width=50, margin_top=50, margin_bottom=50, offset_from_edge=10):
        img_width, img_height = self.size        #identyczne jak self.size
        bar_height = img_height - margin_top - margin_bottom
        hot_colors = np.zeros((bar_height, width, 3), dtype=np.uint8)

        for i in range(bar_height):
            val = 1 - i / bar_height  # Od 1 (góra) do 0 (dół)

            r = int(max(0, min(1.0, val * 3.0 - 0.5)) * 255)
            g = int(max(0, min(1.0, val * 1.5 - 0.25)) * 255)
            b = int(max(0, min(1.0, val * 0.75)) * 255)

            hot_colors[i, :] = [r, g, b]

        colorbar_img = Image.fromarray(hot_colors, 'RGB')

        label_margin = 60*(self.size[0]//1000)  # miejsce na napisy z wartościami
        total_width = img_width + offset_from_edge + width + label_margin
        final_img = Image.new('RGB', (total_width, img_height), color=(255, 255, 255))
        #final_img.paste(img, (0, 0))

        # Tło pod kolorbar
        border = Image.new('RGB', (width + 2, bar_height + 2), (0, 0, 0))
        bar_x = img_width + offset_from_edge
        bar_y = margin_top
        final_img.paste(border, (bar_x - 1, bar_y - 1))
        final_img.paste(colorbar_img, (bar_x, bar_y))

        # Dodawanie podziałki
        draw = ImageDraw.Draw(final_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18*(self.size[0]//1000))
        except:
            font = ImageFont.load_default()

        step = (self.vmax-self.vmin)/8           # co ile wartości pokazywać
        values = np.arange(self.vmin, self.vmax + 1, step)

        for value in values:
            if not (self.vmin <= value <= self.vmax):
                continue
            rel = (value - self.vmin) / (self.vmax - self.vmin)
            y = int(margin_top + (1 - rel) * bar_height)
            draw.line([(bar_x + width, y), (bar_x + width + 10, y)], fill=(0, 0, 0), width=2)
            draw.text((bar_x + width + 12, y - 10), f"{value:.0f}", fill=(0, 0, 0), font=font)
        
        return final_img


    def add_colorbar(self, img, colorbar_img):
        colorbar_img.paste(img, (0, 0))
        return colorbar_img

    # def add_colorbar(self, img, vmin, vmax, width=100):
    #     """Dodaje colorbar do obrazu używając PIL"""
    #     # Utwórz gradient
    #     height = img.height
    #     gradient = np.linspace(1, 0, height).reshape(height, 1)
    #     gradient = np.tile(gradient, (1, width))
    #
    #     # Mapowanie kolorów (hot)
    #     hot_colors = np.zeros((height, width, 3), dtype=np.uint8)
    #     for i in range(height):  # do naprawienia (usunąc fora jakoś)
    #         val = 1 - i / height
    #         r = min(255, max(0, int(val * 4 * 255 - 1.5 * 255)))
    #         g = min(255, max(0, int(val * 2 * 255 - 0.5 * 255)))
    #         b = min(255, max(0, int(val * 2 * 255)))
    #         hot_colors[i, :] = [r, g, b]
    #
    #     # Stwórz finalny obraz
    #     colorbar_img = Image.fromarray(hot_colors, 'RGB')
    #     final_img = Image.new('RGB', (img.width + width, height))
    #     final_img.paste(img, (0, 0))
    #     final_img.paste(colorbar_img, (img.width, 0))
    #
    #     # Dodaj etykiety
    #     draw = ImageDraw.Draw(final_img)
    #
    #     # Ustaw większą czcionkę
    #     font = ImageFont.truetype("arial.ttf", 28)
    #
    #     draw.text((img.width - 90, 10), str(vmax), fill=(0, 0, 0), font=font)
    #     draw.text((img.width - 90, height - 40), str(vmin), fill=(0, 0, 0), font=font)
    #
    #     return final_img

    def create_grid_vao(self, plane, offset):
        """
        Tworzy VAO siatki z automatycznie dobieranym krokiem
        aby zachować proporcje osi w płaszczyźnie
        """
        verts = []

        if plane == 'xy':
            x_values = np.linspace(self.xfliml, self.xflimh, self.grid_steps + 1)
            y_values = np.linspace(self.yfliml, self.yflimh, self.grid_steps + 1)

            # Linie równoległe do Y
            for x in x_values:
                verts.append([x, self.yfliml, offset])
                verts.append([x, self.yflimh, offset])

            # Linie równoległe do X
            for y in y_values:
                verts.append([self.xfliml, y, offset])
                verts.append([self.xflimh, y, offset])

        elif plane == 'yz':
            y_values = np.linspace(self.yfliml, self.yflimh, self.grid_steps + 1)
            z_values = np.linspace(self.zfliml, self.zflimh, self.grid_steps + 1)

            # Linie równoległe do Z
            for y in y_values:
                verts.append([offset, y, self.zfliml])
                verts.append([offset, y, self.zflimh])

            # Linie równoległe do Y
            for z in z_values:
                verts.append([offset, self.yfliml, z])
                verts.append([offset, self.yflimh, z])

        elif plane == 'xz':
            x_values = np.linspace(self.xfliml, self.xflimh, self.grid_steps + 1)
            z_values = np.linspace(self.zfliml, self.zflimh, self.grid_steps + 1)

            # Linie równoległe do Z
            for x in x_values:
                verts.append([x, offset, self.zfliml])
                verts.append([x, offset, self.zflimh])

            # Linie równoległe do X
            for z in z_values:
                verts.append([self.xfliml, offset, z])
                verts.append([self.xflimh, offset, z])

        verts = np.array(verts, dtype='f4')
        vbo = self.ctx.buffer(verts.tobytes())



        return self.ctx.vertex_array(self.grid_prog, [(vbo, '3f4', 'in_position')])

    # def create_single_grid_vao(self, ctx, grid_prog, x_range, y_range, z_fixed, step=1.0):
    #     lines = []
    #
    #     # Linie równoległe do osi X na płaszczyźnie XY, dla stałego z = z_fixed
    #     for y in np.arange(y_range[0], y_range[1] + step, step):
    #         lines.append([x_range[0], y, z_fixed])
    #         lines.append([x_range[1], y, z_fixed])
    #
    #     # Linie równoległe do osi Y na płaszczyźnie XY, dla stałego z = z_fixed
    #     for x in np.arange(x_range[0], x_range[1] + step, step):
    #         lines.append([x, y_range[0], z_fixed])
    #         lines.append([x, y_range[1], z_fixed])
    #
    #     grid_vertices = np.array(lines, dtype='f4')
    #     vbo = ctx.buffer(grid_vertices.tobytes())
    #     vao = ctx.simple_vertex_array(grid_prog, vbo, 'in_position')
    #     return vao

if __name__ == "__main__":
    import numpy as np
    import os

    n = 100_000
    np.random.seed(1)
    xxx = np.random.rand(n) * 60 + 350
    yyy = np.random.rand(n) * 30 + 250
    zzz = np.random.rand(n) * 100 + 100

    vvv = np.random.rand(n) * 25 + 340  # zmienione, tak by pasowało do vmin/vmax

    renderer = ModernGLRenderer(
        xxx=xxx, yyy=yyy, zzz=zzz,
        xfliml=350, xflimh=410,
        yfliml=250, yflimh=280,
        zfliml=100, zflimh=200,
        vmin=340, vmax=365,
        size=(1440,1080),
        point_size=7.0
    )

    output_path = os.path.join(os.getcwd(), 'out_mglhot_v2.png')
    t = time.time()
    renderer.render(vvv, output_path)
    print(f"Render time: {time.time() - t:.4f}s")
    print(f"Image saved to: {output_path}")
