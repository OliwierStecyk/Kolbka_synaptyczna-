import time
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44
import imageio.v3 as iio 


def create_single_grid_vao(ctx, grid_prog, x_range, y_range, z_fixed, step=1.0):
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


def render_moderngl_3d(xxx, yyy, zzz, vvv, 
                      xfliml, xflimh, yfliml, yflimh, zfliml, zflimh, 
                      output_path, size=(1920, 1080), point_size=3.0):
    # 1. Inicjalizacja kontekstu OpenGL (off-screen)
    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
    
    # 2. Przygotowanie danych
    #vmin, vmax = np.min(vvv), np.max(vvv)
    vmin, vmax = (200, 450) # stały zakres skali
    normalized_colors = (vvv - vmin) / (vmax - vmin)
    
    # Konwersja do float32
    vertices = np.column_stack([xxx, yyy, zzz]).astype('f4')
    colors = normalized_colors.astype('f4')

    vertex_shader = f"""
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    in float in_color;
    out float v_color;
    void main() {{
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
        gl_PointSize = {point_size};
    }}
    """

    fragment_shader = """
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

    grid_vertex_shader = """
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    """
    grid_fragment_shader = """
    #version 330
    out vec4 fragColor;
    void main() {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // czarna siatka
    }
    """

    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    grid_prog = ctx.program(vertex_shader=grid_vertex_shader, fragment_shader=grid_fragment_shader)

    vbo_verts = ctx.buffer(vertices.tobytes())
    vbo_colors = ctx.buffer(colors.tobytes())
    
    vao = ctx.vertex_array(
        prog,
        [
            (vbo_verts, '3f4', 'in_position'),
            (vbo_colors, '1f4', 'in_color')
        ]
    )
    # 6. Macierze transformacji
    center = np.array([np.mean([xfliml, xflimh]), 
                      np.mean([yfliml, yflimh]), 
                      np.mean([zfliml, zflimh])])
    
    radius = max(xflimh-xfliml, yflimh-yfliml, zflimh-zfliml) * 1.5
    
    proj = Matrix44.perspective_projection(45.0, size[0]/size[1], 0.1, radius*2)
    view = Matrix44.look_at(
        (center[0], center[1], center[2] + radius),
        center,
        (0, 1, 0)
    )
    mvp = proj * view
    
    # 7. Renderowanie
    prog['mvp'].write(mvp.astype('f4').tobytes())
    grid_prog['mvp'].write(mvp.astype('f4').tobytes())

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture(size, 4)],
        depth_attachment=ctx.depth_texture(size)
    )       #dobrze
    fbo.use()
    fbo.clear(1.0, 1.0, 1.0, 1.0)  # Białe tło
    #prog['vmin'].value = vmin
    #prog['vmax'].value = vmax
    
    vao.render(moderngl.POINTS)

    grid_vao = create_single_grid_vao(
        ctx, grid_prog,
        x_range=(xfliml, xflimh),
        y_range=(yfliml, yflimh),
        z_fixed=zfliml,  # jedna płaska kratka na dolnej granicy osi Z
        step=(xflimh - xfliml) / 10.0
    )
    grid_vao.render(moderngl.LINES)

    img_data = fbo.read(components=3)
    img = Image.frombytes('RGB', size, img_data)
    # 9. Dodanie colorbar
    img = add_colorbar(img, vmin, vmax)

    iio.imwrite(output_path, image = img,
               compression=6)
   # img.save(output_path) # wąskie gardło 0.1s

    vao.release()
    vbo_verts.release()
    vbo_colors.release()
    prog.release()
    fbo.release()
    ctx.release()


def add_colorbar(img, vmin, vmax, width=100):
    """Dodaje colorbar do obrazu używając PIL"""
    # Utwórz gradient
    height = img.height
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.tile(gradient, (1, width))
    
    # Mapowanie kolorów (hot)
    hot_colors = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        val = 1 - i/height
        r = min(255, max(0, int(val * 4 * 255 - 1.5 * 255)))
        g = min(255, max(0, int(val * 2 * 255 - 0.5 * 255)))
        b = min(255, max(0, int(val * 2 * 255)))
        hot_colors[i, :] = [r, g, b]
    
    # Stwórz finalny obraz
    colorbar_img = Image.fromarray(hot_colors, 'RGB')
    final_img = Image.new('RGB', (img.width + width, height), 'white')
    final_img.paste(img, (0, 0))
    final_img.paste(colorbar_img, (img.width, 0))
    
    # Dodaj etykiety
    draw = ImageDraw.Draw(final_img)

    # Ustaw większą czcionkę
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font = ImageFont.load_default()

    draw.text((img.width - 90, 10), str(vmax), fill=(0, 0, 0), font=font)
    draw.text((img.width - 90, height-40), str(vmin), fill=(0, 0, 0), font=font)
    
    return final_img

# Przykład użycia:
if __name__ == "__main__":
    # Generowanie danych testowych
    n = 3102  # 100k punktów
    xxx = np.random.rand(n) * 10
    yyy = np.random.rand(n) * 10
    zzz = np.random.rand(n) * 10
    vvv = np.random.rand(n) * 25 + 340  # zakres ~340-365
    vvv1 = np.random.rand(n) * 25 + 220  # zakres ~340-365
    t0 = time.time()
    # Renderowanie
    render_moderngl_3d(
        xxx, yyy, zzz, vvv,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl.png"
    )
    print(f'Czas renderowania: {time.time()-t0:.7f} s')
    t3 = time.time()
    render_moderngl_3d(
        xxx, yyy, zzz, vvv1,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl1.png"
    )
    print(f'Czas renderowania 2: {time.time()-t3:.7f} s')
