import pygame
from pg_utils import *

# Colors Setup
colors = Color()
colors.init_colors()

# PyGame Setup
pygame.init()

SCREEN = pygame.display.set_mode(RESOLUTION, pygame.FULLSCREEN if FULLSCREEN else 0)
pygame.display.set_caption(WINDOW_NAME)
pygame.display.set_icon(pygame.image.load(ICON_LOCATION))

clock = pygame.time.Clock()
delta_time = 0.0

FOV = 500

arial = pygame.font.SysFont("arial", 48)
arial_small = pygame.font.SysFont("arial", 24)

draw_gizmos = True
draw_overlays = True
draw_debug = True
gizmos_size = 50


# This whole project revolves around this function.
# Weak Perspective Projection - A type of projection that is not weak, but powerfull. Basically, it projects 3D shapes into a 2D plane that is some distance away from the camera (that distance is the focal length).

# The formulas go like this:

#       x * f
# Px = -------
#       z + f

#       y * f
# Py = -------
#       z + f

# Px and Py are X projected and Y projected, f is the focal length variable A.K.A FOV
# The focal length variable is so that objects appear smaller the further away they are, which is, you know, realistic.


def wpp(vector: Vector3) -> Vector2:
    near_clip = 0.1  # or any small positive number to prevent distortion
    x_projected = (vector.x * FOV) / max(vector.z + FOV, near_clip)
    y_projected = (vector.y * FOV) / max(vector.z + FOV, near_clip)
    return Vector2(x_projected, y_projected)


class LightSource:
    def __init__(self, position: Vector3, color: tuple, intensity=1.0, range=1000.0):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.range = range

    def calculate_intensity(self, normal: Vector3, point: Vector3) -> float:
        # Calculate the direction of the light
        light_direction = (self.position - point).normalize()

        # Dot product between the light direction and the face normal
        dot_intensity = max(normal.dot(light_direction), 0)

        # Calculate the distance from the light source to the point
        distance = (self.position - point).magnitude()

        # Apply attenuation based on distance
        attenuation = max(1 - (distance / self.range), 0)

        # Calculate final intensity
        final_intensity = self.intensity * dot_intensity * attenuation

        return final_intensity

    # Basic gizmos drawing function to visualize where a light source is located
    def gizmos(self):
        self.light_gizmos = pygame.image.load("gizmos/light_white.png")
        self.light_gizmos = pygame.transform.scale(
            self.light_gizmos, (gizmos_size, gizmos_size)
        )

        self.size = gizmos_size

        self.gizmos_rect = (
            wpp(self.position).x + WIDTH // 2 - self.size,
            -wpp(self.position).y + HEIGHT // 2 - self.size,
            self.size,
            self.size,
        )

        SCREEN.blit(self.light_gizmos, self.gizmos_rect)


class Cube:
    def __init__(
        self,
        position: Vector3,
        rotation: Vector3,
        scale: Vector3,
        color: tuple,
        fill_in=False,
        draw_wireframe=True,
        wireframe_width=2,
        outline_color=colors.WHITE,
    ):
        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.color = color
        self.outline_color = outline_color

        self.fill_in = fill_in
        self.draw_wireframe = draw_wireframe
        self.wireframe_width = wireframe_width

        # This list defines the cube, we write an algorithm that can change this into an ico_sphere and not change anything else (besides the splitting from quads to triangles part because an icosphere is already made of triangle sides)
        self.points = [
            Vector3(-1, -1, -1),
            Vector3(-1, -1, 1),
            Vector3(1, -1, -1),
            Vector3(1, -1, 1),
            Vector3(-1, 1, -1),
            Vector3(-1, 1, 1),
            Vector3(1, 1, -1),
            Vector3(1, 1, 1),
        ]

        # Face indexes
        self.faces = [
            [0, 1, 5, 4],  # LEFT
            [1, 3, 7, 5],  # BACK
            [3, 2, 6, 7],  # RIGHT
            [2, 0, 4, 6],  # FRONT
            [0, 2, 3, 1],  # BOTTOM
            [4, 5, 7, 6],  # TOP
        ]

    def rotate_x(self, vector: Vector3, angle: float):
        rad = math.radians(angle)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)

        # Using rotation matrix and multiplying the values correctly
        return Vector3(
            vector.x,
            vector.y * cos_theta - vector.z * sin_theta,
            vector.y * sin_theta + vector.z * cos_theta,
        )

    def rotate_y(self, vector: Vector3, angle: float):
        rad = math.radians(angle)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)

        # Using rotation matrix and multiplying the values correctly
        return Vector3(
            vector.x * cos_theta - vector.z * sin_theta,
            vector.y,
            vector.x * sin_theta + vector.z * cos_theta,
        )

    def rotate_z(self, vector: Vector3, angle: float):
        rad = math.radians(angle)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)

        # Using rotation matrix and multiplying the values correctly
        return Vector3(
            vector.x * cos_theta - vector.y * sin_theta,
            vector.x * sin_theta + vector.y * cos_theta,
            vector.z,
        )

    def calculate_normal(self, p1: Vector3, p2: Vector3, p3: Vector3) -> Vector3:
        # Standard normal calculation
        u = p2 - p1
        v = p3 - p1
        normal = u.cross(v).normalize()
        return normal

    def calculate_face_depth(self, vertices: list[Vector3]) -> float:
        # Sum the Z coordinates from all vertices and divide by amount of vertices
        z_total = sum([v.z for v in vertices])
        return z_total / len(vertices)

    def render(self, light_sources: list):
        transformed_points = []

        # Transform square vertices
        for point in self.points:
            # Scale points
            translated_point = point * self.scale

            # Rotate points
            rotated_point = self.rotate_x(
                self.rotate_y(
                    self.rotate_z(translated_point, self.rotation.z), self.rotation.y
                ),
                self.rotation.x,
            )
            # Calculate position of points based of the rotated and translated point variables
            final_point = rotated_point + self.position
            transformed_points.append(final_point)

        faces_to_render = []

        for face in self.faces:
            v1, v2, v3, v4 = [transformed_points[i] for i in face]

            # Calculate normal to determine if the face is visible (back-face culling)
            normal = self.calculate_normal(v1, v2, v3)
            view_vector = (v1 - Vector3(0, 0, -FOV)).normalize()

            if normal.dot(view_vector) < 0:  # Face is visible
                # Break face into two triangles
                triangles = [
                    [v1, v2, v3],
                    [v1, v3, v4],
                ]

                for triangle in triangles:
                    # Get normal direction of triangle
                    normal = self.calculate_normal(
                        triangle[0], triangle[1], triangle[2]
                    )

                    # Calculate color intensity with the LightSource.calculate_intensity() function
                    intensity = 0
                    for light in light_sources:
                        intensity += light.calculate_intensity(normal, triangle[0])

                    # Now apply the intensity to self.color and store values in tuple so we can access individual colors later
                    shaded_color = tuple(
                        int(c * intensity) for c in self.color.get_tup()
                    )

                    # Calculate depth for sorting
                    depth = self.calculate_face_depth(triangle)

                    # Store each triangle with its depth and color
                    faces_to_render.append((triangle, shaded_color, depth))

        # Sort triangles by depth (draw the farthest first)
        faces_to_render.sort(key=lambda item: item[2], reverse=True)

        # Render triangles (better than squares because its just how it is)
        for triangle, color, _ in faces_to_render:

            # Use the weak perspective projection function from before
            vertices_2d = [wpp(v).get_tup() for v in triangle]

            # Clamp color because the color might be over 255 with the lighting method we are using.
            color = (
                clamp(color[0], 0, 255),
                clamp(color[1], 0, 255),
                clamp(color[2], 0, 255),
            )

            # Finally draw triangles
            pygame.draw.polygon(
                SCREEN,
                color,
                [(v[0] + WIDTH // 2, -v[1] + HEIGHT // 2) for v in vertices_2d],
            )

        if self.draw_wireframe:
            for face in self.faces:
                for i in range(4):
                    # wpp = Weak Perspective Projection
                    point1 = wpp(transformed_points[face[i]]).get_tup()
                    point2 = wpp(transformed_points[face[(i + 1) % 4]]).get_tup()

                    pygame.draw.line(
                        SCREEN,
                        self.outline_color.get_tup(),
                        (point1[0] + WIDTH // 2, -point1[1] + HEIGHT // 2),
                        (point2[0] + WIDTH // 2, -point2[1] + HEIGHT // 2),
                        self.wireframe_width,
                    )


# This is to correctly render the cubes in the correct order based on the distance from camera
def sort_based_on_distance(lst: list):
    distances = []

    for i in range(len(lst)):
        vector = lst[i].position - Vector3(FOV, FOV, FOV)
        item = Vector3(vector.x, vector.y, vector.z).magnitude()
        distances.append(item)
        print(distances[i])

    for i in range(len(lst) - 1, 0, -1):
        swapped = False

        for j in range(i):
            if distances[j] > distances[j + 1]:
                swapped = True
                distances[j], distances[j + 1] = distances[j + 1], distances[j]
                lst[j], lst[j + 1] = lst[j + 1], lst[j]

        if not swapped:
            continue

    return lst


# Setup the objects
def setup():
    light = LightSource(Vector3(0, 0, 0), colors.WHITE, intensity=1.0, range=3000.0)

    objects = []
    lights = [light]

    for i in range(10):
        for j in range(10):
            c = Cube(
                Vector3(i * 500 - 2000, j * 500 - 2000, 400),
                Vector3(0, 0, 0),
                Vector3(100, 100, 100),
                colors.random(),
                True,
                False,
            )

            objects.append(c)

    c = Cube(
        Vector3(0, 0, 300),
        Vector3(0, 0, 0),
        Vector3(150, 150, 150),
        colors.WHITE,
        True,
        False,
    )
    objects.append(c)

    return objects, lights


def main():
    global delta_time
    global draw_gizmos, draw_overlays, draw_debug

    objects, light_sources = setup()

    fps_list = []

    running = True
    get_ticks_last_frame = 0.0
    elapsed_time = 0.0
    frames_elapsed = 0

    while running == True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_g and draw_overlays:
                    draw_gizmos = not draw_gizmos
                if event.key == pygame.K_o:
                    draw_overlays = not draw_overlays
                if event.key == pygame.K_d and draw_overlays:
                    draw_debug = not draw_debug

        elapsed_time += delta_time
        frames_elapsed += 1

        objects = sort_based_on_distance(objects)

        SCREEN.fill(colors.BLACK.get_tup())

        # Cube rotation
        for i in range(len(objects)):
            objects[i].rotation.x += math.sin(elapsed_time + i)
            objects[i].rotation.y += math.cos(elapsed_time + i)
            objects[i].rotation.z += (
                objects[i].rotation.x * math.sin(elapsed_time * i) / 1000
            )

        light_sources[0].position.x = -math.cos(elapsed_time) * 800
        light_sources[0].position.y = math.sin(elapsed_time) * 800

        # FPS calculation
        fps = clock.get_fps()
        fps_list.append(fps)

        # Average out all recorded frames with their FPS
        average_fps = sum(fps_list) / len(fps_list)

        # We dont want a really big list otherwise it might start to lag
        if len(fps_list) > 50_000:
            fps_list.pop(1)

        # Render objects
        for obj in objects:
            obj.render(light_sources)

        # Draw gizmos and other overlays
        if draw_overlays:
            if draw_gizmos:
                for light in light_sources:
                    light.gizmos()

            fps_text = Text(
                f"FPS: {fps:.2f}", arial, colors.WHITE.get_tup(), (50, 50), True, True
            )
            fps_text.draw(SCREEN, "topleft")

            average_fps_text = Text(
                f"AVERAGE FPS: {average_fps:.2f}",
                arial,
                colors.WHITE.get_tup(),
                (50, 100),
                True,
                True,
            )
            average_fps_text.draw(SCREEN, "topleft")

            if draw_debug:
                fps_list_text = Text(
                    f"DEBUG: FPS LIST SIZE: {len(fps_list)}",
                    arial_small,
                    colors.DARK_GRAY.get_tup(),
                    (50, HEIGHT - 50),
                    True,
                )
                fps_list_text.draw(SCREEN, "bottomleft")

                frames_elapsed_text = Text(
                    f"DEBUG: FRAMES ELAPSED: {frames_elapsed}",
                    arial_small,
                    colors.DARK_GRAY.get_tup(),
                    (50, HEIGHT - 82),
                    True,
                )
                frames_elapsed_text.draw(SCREEN, "bottomleft")

                delta_time_text = Text(
                    f"DEBUG: DELTA TIME: {delta_time*1000:1.0f} ms",
                    arial_small,
                    colors.DARK_GRAY.get_tup(),
                    (50, HEIGHT - 114),
                    True,
                )
                delta_time_text.draw(SCREEN, "bottomleft")

        pygame.display.flip()

        get_ticks_last_frame, delta_time = manage_frame_rate(
            clock, get_ticks_last_frame
        )

    pygame.quit()


if __name__ == "__main__":
    main()
