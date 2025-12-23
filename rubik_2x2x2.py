import os
import pickle
import random
from datetime import datetime

import cv2
import numpy as np
from ursina import (
    Entity,
    Text,
    Ursina,
    Vec3,
    camera,
    color,
    destroy,
    distance,
    held_keys,
    invoke,
    lerp,
    mouse,
    scene,
    time,
    window,
)


class rubik_2x2x2:
    def __init__(self, save_path="", shuffle_num=50, phase=4):
        self.phase = phase
        if save_path:
            with open(save_path, "rb") as f:
                self.cube = pickle.load(f)
        else:
            # 2x2x2 flattened map (6x6 grid)
            # Layout:
            #       [U]
            #    [L][F][R]
            #       [D][B]
            self.cube = np.zeros((6, 6), dtype=np.uint8)

            # 1:White(U), 2:Orange(L), 3:Green(F), 4:Red(R), 5:Yellow(D), 6:Blue(B)
            self.cube[0:2, 2:4] = 1  # Up
            self.cube[2:4, 0:2] = 2  # Left
            self.cube[2:4, 2:4] = 3  # Front
            self.cube[2:4, 4:6] = 4  # Right
            self.cube[4:6, 2:4] = 5  # Down
            self.cube[4:6, 4:6] = 6  # Back

            self.target = self.cube.copy()

            self.shuffle()

    def shuffle(self, shuffle_num=50):
        manipulate_num = 12
        for i in range(shuffle_num):
            rand = int(random.random() * manipulate_num)
            moves = [self.R, self.Ri, self.Li, self.L, self.Ui, self.U,
                     self.D, self.Di, self.F, self.Fi, self.Bi, self.B]
            moves[rand]()

    # --- Rotation Logic (Hardcoded for 2x2) ---
    # R: Right face CW
    def R(self):
        y_ind = [2, 3, 4, 5, 5, 4, 0, 1]
        x_ind = [3, 3, 3, 3, 5, 5, 3, 3]
        y_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        x_pri = [3, 3, 3, 3, 3, 3, 5, 5]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 4:6] = np.rot90(self.cube[2:4, 4:6], k=-1)

    def Ri(self):
        y_ind = [5, 4, 0, 1, 2, 3, 4, 5]
        x_ind = [5, 5, 3, 3, 3, 3, 3, 3]
        y_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        x_pri = [3, 3, 3, 3, 3, 3, 5, 5]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 4:6] = np.rot90(self.cube[2:4, 4:6], k=1)

    def Li(self):
        y_ind = [2, 3, 4, 5, 5, 4, 0, 1]
        x_ind = [2, 2, 2, 2, 4, 4, 2, 2]
        y_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        x_pri = [2, 2, 2, 2, 2, 2, 4, 4]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 0:2] = np.rot90(self.cube[2:4, 0:2], k=1)

    def L(self):
        y_ind = [5, 4, 0, 1, 2, 3, 4, 5]
        x_ind = [4, 4, 2, 2, 2, 2, 2, 2]
        y_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        x_pri = [2, 2, 2, 2, 2, 2, 4, 4]

        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 0:2] = np.rot90(self.cube[2:4, 0:2], k=-1)

    def U(self):
        y_ind = [2, 2, 2, 2, 4, 4, 2, 2]
        x_ind = [2, 3, 4, 5, 5, 4, 0, 1]
        y_pri = [2, 2, 2, 2, 2, 2, 4, 4]
        x_pri = [0, 1, 2, 3, 4, 5, 5, 4]

        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[0:2, 2:4] = np.rot90(self.cube[0:2, 2:4], k=-1)

    def Ui(self):
        y_ind = [4, 4, 2, 2, 2, 2, 2, 2]
        x_ind = [5, 4, 0, 1, 2, 3, 4, 5]
        y_pri = [2, 2, 2, 2, 2, 2, 4, 4]
        x_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[0:2, 2:4] = np.rot90(self.cube[0:2, 2:4], k=1)

    def D(self):
        y_ind = [5, 5, 3, 3, 3, 3, 3, 3]
        x_ind = [5, 4, 0, 1, 2, 3, 4, 5]
        y_pri = [3, 3, 3, 3, 3, 3, 5, 5]
        x_pri = [0, 1, 2, 3, 4, 5, 5, 4]

        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[4:6, 2:4] = np.rot90(self.cube[4:6, 2:4], k=-1)

    def Di(self):
        y_ind = [3, 3, 3, 3, 5, 5, 3, 3]
        x_ind = [2, 3, 4, 5, 5, 4, 0, 1]
        y_pri = [3, 3, 3, 3, 3, 3, 5, 5]
        x_pri = [0, 1, 2, 3, 4, 5, 5, 4]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[4:6, 2:4] = np.rot90(self.cube[4:6, 2:4], k=1)

    def F(self):
        y_ind = [3, 2, 1, 1, 2, 3, 4, 4]
        x_ind = [1, 1, 2, 3, 4, 4, 3, 2]
        y_pri = [1, 1, 2, 3, 4, 4, 3, 2]
        x_pri = [2, 3, 4, 4, 3, 2, 1, 1]

        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 2:4] = np.rot90(self.cube[2:4, 2:4], k=-1)

    def Fi(self):
        y_ind = [2, 3, 4, 4, 3, 2, 1, 1]
        x_ind = [4, 4, 3, 2, 1, 1, 2, 3]
        y_pri = [1, 1, 2, 3, 4, 4, 3, 2]
        x_pri = [2, 3, 4, 4, 3, 2, 1, 1]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[2:4, 2:4] = np.rot90(self.cube[2:4, 2:4], k=1)

    def B(self):
        y_ind = [2, 3, 5, 5, 3, 2, 0, 0]
        x_ind = [5, 5, 3, 2, 0, 0, 2, 3]
        y_pri = [0, 0, 2, 3, 5, 5, 3, 2]
        x_pri = [2, 3, 5, 5, 3, 2, 0, 0]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[4:6, 4:6] = np.rot90(self.cube[4:6, 4:6], k=1)

    def Bi(self):
        y_ind = [3, 2, 0, 0, 2, 3, 5, 5]
        x_ind = [0, 0, 2, 3, 5, 5, 3, 2]
        y_pri = [0, 0, 2, 3, 5, 5, 3, 2]
        x_pri = [2, 3, 5, 5, 3, 2, 0, 0]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[4:6, 4:6] = np.rot90(self.cube[4:6, 4:6], k=-1)

    def get_state(self):
        # 9x9の配列全体をバイト列化して一意なIDとする
        copy_cube = self.cube.copy()
        return copy_cube.tobytes()

    def update(self, rotate_index):  # noqa: C901
        if rotate_index == 0:
            self.R()
        elif rotate_index == 1:
            self.Ri()
        elif rotate_index == 2:
            self.Li()
        elif rotate_index == 3:
            self.L()
        elif rotate_index == 4:
            self.Ui()
        elif rotate_index == 5:
            self.U()
        elif rotate_index == 6:
            self.D()
        elif rotate_index == 7:
            self.Di()
        elif rotate_index == 8:
            self.F()
        elif rotate_index == 9:
            self.Fi()
        elif rotate_index == 10:
            self.Bi()
        elif rotate_index == 11:
            self.B()

    def show_rubik_2Dmap(self):
        height, width = self.cube.shape
        bgr_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Colors
        colors = {
            1: (255, 255, 255),  # White
            2: (40, 117, 232),  # Orange
            3: (0, 128, 0),     # Green
            4: (28, 0, 198),    # Red
            5: (28, 211, 251),  # Yellow
            6: (153, 51, 0)     # Blue
        }
        for k, v in colors.items():
            bgr_image[self.cube == k] = np.array(v)

        bgr_image = cv2.resize(bgr_image, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("rubic_2Dmap", bgr_image)

    def save_rubik_2Dmap(self):
        try:
            os.mkdir(f"savefiles/{datetime.now():%Y%m%d}")
        except FileExistsError:
            pass
        with open(f"savefiles/{datetime.now():%Y%m%d}/{datetime.now():%H%M%S}.pkl", "wb") as f:
            pickle.dump(self.cube, f)
        print("Successfully saved 2D Rubik map")


class RubikCubeCamera(Entity):
    def __init__(self, initial_position=(6, 6, -10), rotate_speed=130, return_speed=10, save_path="", **kwargs):
        super().__init__(**kwargs)
        self.app = Ursina()
        # Scale adjusted for 2x2
        self.target = Entity(model='cube', scale=2 / np.sqrt(2), color=color.black)
        self.rotator = Entity()
        self.initial_position = initial_position
        self.rotate_speed = rotate_speed
        self.return_speed = return_speed
        self.cubes = []
        self.action_mode = False
        self.colors = {
            0: color.clear, 1: color.white, 2: color.orange, 3: color.green,
            4: color.red, 5: color.yellow, 6: color.azure
        }

        self.ribik = rubik_2x2x2(save_path=save_path)
        self.ribik.show_rubik_2Dmap()
        self.draw_ursina_cube()

        self.pivot = Entity(position=self.target.position)
        camera.parent = self.pivot
        camera.position = self.initial_position
        camera.rotation_z = -5
        camera.look_at(self.pivot)
        self.default_rotation = self.pivot.rotation

        # Improved HUD Text
        text = (
            "2x2 Pocket Cube\n"
            "----------------\n"
            "Right Click : Orbit\n"
            "Release     : Reset\n"
            "S Key       : Save\n\n"
            "Controls (Numpad):\n"
            " R: * | R': 3\n"
            " L: 2 | L': /\n"
            " U: 7 | U': -\n"
            " D: + | D': 4\n"
            " F: 8 | F': 9\n"
            " B: 5 | B': 6"
        )
        Text(text=text, position=(-0.85, 0.45), origin=(-0.5, 0.5), scale=1.0, line_height=1.1)

    def update(self):
        if held_keys['right mouse']:
            self.pivot.rotation_y += mouse.velocity[0] * self.rotate_speed
            self.pivot.rotation_x -= mouse.velocity[1] * self.rotate_speed * 2
        else:
            self.pivot.rotation = lerp(self.pivot.rotation, self.default_rotation, time.dt * self.return_speed)

    def rotate_side(self, side_name):  # noqa: C901
        if self.action_mode:
            return
        self.action_mode = True
        self.rotator.rotation = (0, 0, 0)

        # Parenting for 2x2: Pivot is at 0, cubes are at +/- 0.5
        # Threshold > 0 or < 0 works correctly
        for e in self.cubes:
            if side_name == '*' and e.world_x > 0:
                e.world_parent = self.rotator
            elif side_name == '3' and e.world_x > 0:
                e.world_parent = self.rotator
            elif side_name == '2' and e.world_x < 0:
                e.world_parent = self.rotator
            elif side_name == '/' and e.world_x < 0:
                e.world_parent = self.rotator
            elif side_name == '7' and e.world_y > 0:
                e.world_parent = self.rotator
            elif side_name == '-' and e.world_y > 0:
                e.world_parent = self.rotator
            elif side_name == '+' and e.world_y < 0:
                e.world_parent = self.rotator
            elif side_name == '4' and e.world_y < 0:
                e.world_parent = self.rotator
            elif side_name == '5' and e.world_z > 0:
                e.world_parent = self.rotator  # B
            elif side_name == '6' and e.world_z > 0:
                e.world_parent = self.rotator  # B
            elif side_name == '9' and e.world_z < 0:
                e.world_parent = self.rotator  # F (Front is negative Z)
            elif side_name == '8' and e.world_z < 0:
                e.world_parent = self.rotator  # F

        # Animation axis
        if side_name in ['*', '3', '2', '/']:
            axis = 'rotation_x'
        elif side_name in ['7', '-', '+', '4']:
            axis = 'rotation_y'
        elif side_name in ['5', '9', '6', '8']:
            axis = 'rotation_z'

        # Apply logic
        mapping = {
            '*': self.ribik.R, '3': self.ribik.Ri,
            '2': self.ribik.L, '/': self.ribik.Li,
            '7': self.ribik.U, '-': self.ribik.Ui,
            '+': self.ribik.D, '4': self.ribik.Di,
            '8': self.ribik.F, '9': self.ribik.Fi,
            '5': self.ribik.B, '6': self.ribik.Bi
        }
        if side_name in mapping:
            mapping[side_name]()

        self.ribik.show_rubik_2Dmap()

        # Rotation Angle
        angle = 90 if side_name in ['*', '/', '7', '9', '4', '6'] else -90
        self.rotator.animate(axis, angle, duration=0.1)
        invoke(self.reset_structure, delay=0.12)

    def reset_structure(self):
        for e in self.cubes:
            e.world_parent = scene
        self.action_mode = False

    def input(self, key):
        k = key.upper()
        if k in ['*', '2', '/', '3', '7', '-', '+', '4', '8', '9', '6', '5']:
            self.rotate_side(k)
        elif k == "S":
            self.ribik.save_rubik_2Dmap()

    def draw_ursina_cube(self):
        cube_state = self.ribik.cube

        # 2x2x2 means 8 cubes. Positions are -0.5 and 0.5
        positions = [-0.5, 0.5]

        for x in positions:
            for y in positions:
                for z in positions:
                    c = Entity(model='cube', color=color.black, position=(x, y, z), scale=0.98)
                    self.cubes.append(c)

        # Config adjusted for 2x2 indices
        # Grid layout: U[0:2, 2:4], L[2:4, 0:2], F[2:4, 2:4], R[2:4, 4:6], D[4:6, 2:4], B[4:6, 4:6]
        # Pos mapping: r(row 0-1), c(col 0-1) -> World Space
        faces_config = [
            # Top (U): y=0.5
            {'slice': (0, 2, 2, 4), 'pos': lambda r, c: Vec3(c - 0.5, 0.5, 0.5 - r), 'rot': (90, 0, 0)},
            # Left (L): x=-0.5
            {'slice': (2, 4, 0, 2), 'pos': lambda r, c: Vec3(-0.5, 0.5 - r, 0.5 - c), 'rot': (0, 90, 0)},
            # Front (F): z=-0.5
            {'slice': (2, 4, 2, 4), 'pos': lambda r, c: Vec3(c - 0.5, 0.5 - r, -0.5), 'rot': (0, 0, 0)},
            # Right (R): x=0.5
            {'slice': (2, 4, 4, 6), 'pos': lambda r, c: Vec3(0.5, 0.5 - r, c - 0.5), 'rot': (0, -90, 0)},
            # Bottom (D): y=-0.5
            {'slice': (4, 6, 2, 4), 'pos': lambda r, c: Vec3(c - 0.5, -0.5, r - 0.5), 'rot': (-90, 0, 0)},
            # Back (B): z=0.5
            {'slice': (4, 6, 4, 6), 'pos': lambda r, c: Vec3(c - 0.5, 0.5 - r, 0.5), 'rot': (0, 180, 0)},
        ]

        for config in faces_config:
            rs, re, cs, ce = config['slice']
            face_data = cube_state[rs:re, cs:ce]
            target_rot = Vec3(*config['rot'])

            for r in range(2):
                for c in range(2):
                    color_code = face_data[r, c]
                    if color_code == 0:
                        continue

                    target_pos = config['pos'](r, c)

                    for cube_entity in self.cubes:
                        if distance(cube_entity.position, target_pos) < 0.1:
                            sticker = Entity(
                                parent=cube_entity,
                                model='quad',
                                color=self.colors[color_code],
                                scale=0.9,
                                texture='white_cube'
                            )
                            sticker.world_position = target_pos
                            sticker.world_rotation = target_rot
                            sticker.world_position += sticker.back * 0.6  # Slightly closer for smaller cubes
                            break

    def run_app(self):
        window.color = color.dark_gray
        window.title = '2x2 Rubik\'s Cube'
        self.app.run()


if __name__ == '__main__':
    rcc = RubikCubeCamera()
    rcc.run_app()
