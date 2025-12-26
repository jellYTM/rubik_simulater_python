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


class rubik_3x3x3:
    def __init__(self, save_path="", shuffle_num=50):
        self.save_path = ""
        if save_path:
            with open(save_path, "rb") as f:
                self.cube = pickle.load(f)
        else:
            # 3x3x3 flattened map (9x9 grid)
            # Layout:
            #       [U]
            #    [L][F][R]
            #       [D][B]
            self.cube = np.zeros((9, 9), dtype=np.uint8)

            # 1:White(U), 2:Orange(L), 3:Green(F), 4:Red(R), 5:Yellow(D), 6:Blue(B)
            self.cube[0:3, 3:6] = 1  # Up
            self.cube[3:6, 0:3] = 2  # Left
            self.cube[3:6, 3:6] = 3  # Front
            self.cube[3:6, 6:9] = 4  # Right
            self.cube[6:9, 3:6] = 5  # Down
            self.cube[6:9, 6:9] = 6  # Back

            self.shuffle(shuffle_num)

    def shuffle(self, shuffle_num=50):
        manipulate_num = 12
        for i in range(shuffle_num):
            rand = int(random.random() * manipulate_num)
            moves = [self.R, self.Ri, self.Li, self.L, self.Ui, self.U,
                     self.D, self.Di, self.F, self.Fi, self.Bi, self.B]
            moves[rand]()

    # --- Rotation Logic (Hardcoded for 3x3) ---
    def R(self):
        y_ind = [3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2]
        x_ind = [5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5]
        y_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        x_pri = [5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 6:9] = np.rot90(self.cube[3:6, 6:9], k=-1)

    def Ri(self):
        y_ind = [8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        x_ind = [8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        y_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        x_pri = [5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 6:9] = np.rot90(self.cube[3:6, 6:9], k=1)

    def Li(self):
        y_ind = [3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2]
        x_ind = [3, 3, 3, 3, 3, 3, 6, 6, 6, 3, 3, 3]
        y_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        x_pri = [3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 0:3] = np.rot90(self.cube[3:6, 0:3], k=1)

    def L(self):
        y_ind = [8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        x_ind = [6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        y_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        x_pri = [3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 0:3] = np.rot90(self.cube[3:6, 0:3], k=-1)

    def Ui(self):
        y_ind = [6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        x_ind = [8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        y_pri = [3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6]
        x_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[0:3, 3:6] = np.rot90(self.cube[0:3, 3:6], k=1)

    def U(self):
        y_ind = [3, 3, 3, 3, 3, 3, 6, 6, 6, 3, 3, 3]
        x_ind = [3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2]
        y_pri = [3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6]
        x_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[0:3, 3:6] = np.rot90(self.cube[0:3, 3:6], k=-1)

    def D(self):
        y_ind = [8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        x_ind = [8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        y_pri = [5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8]
        x_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[6:9, 3:6] = np.rot90(self.cube[6:9, 3:6], k=-1)

    def Di(self):
        y_ind = [5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5]
        x_ind = [3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2]
        y_pri = [5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8]
        x_pri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[6:9, 3:6] = np.rot90(self.cube[6:9, 3:6], k=1)

    def F(self):
        y_ind = [5, 4, 3, 2, 2, 2, 3, 4, 5, 6, 6, 6]
        x_ind = [2, 2, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3]
        y_pri = [2, 2, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3]
        x_pri = [3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 2, 2]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 3:6] = np.rot90(self.cube[3:6, 3:6], k=-1)

    def Fi(self):
        y_ind = [3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 2, 2]
        x_ind = [6, 6, 6, 5, 4, 3, 2, 2, 2, 3, 4, 5]
        y_pri = [2, 2, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3]
        x_pri = [3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 2, 2]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[3:6, 3:6] = np.rot90(self.cube[3:6, 3:6], k=1)

    def Bi(self):
        y_ind = [5, 4, 3, 0, 0, 0, 3, 4, 5, 8, 8, 8]
        x_ind = [0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3]
        y_pri = [0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3]
        x_pri = [3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[6:9, 6:9] = np.rot90(self.cube[6:9, 6:9], k=-1)

    def B(self):
        y_ind = [3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0]
        x_ind = [8, 8, 8, 5, 4, 3, 0, 0, 0, 3, 4, 5]
        y_pri = [0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3]
        x_pri = [3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0]
        self.cube[y_pri, x_pri] = self.cube[y_ind, x_ind]
        self.cube[6:9, 6:9] = np.rot90(self.cube[6:9, 6:9], k=1)

    def get_state(self):
        # 9x9の配列全体をバイト列化して一意なIDとする
        copy_cube = self.cube.copy()
        return copy_cube.tobytes()

    def update(self, rotate_index):
        moves = [self.R, self.Ri, self.Li, self.L, self.Ui, self.U,
                    self.D, self.Di, self.F, self.Fi, self.Bi, self.B]
        moves[rotate_index]()

    def show_rubik_2Dmap(self):
        height, width = self.cube.shape
        bgr_image = np.zeros((height, width, 3), dtype=np.uint8)
        bgr_image[self.cube == 1] = np.array((255, 255, 255))  # white
        bgr_image[self.cube == 2] = np.array((40, 117, 232))  # orange
        bgr_image[self.cube == 3] = np.array((0, 128, 0))  # green
        bgr_image[self.cube == 4] = np.array((28, 0, 198))  # red
        bgr_image[self.cube == 5] = np.array((28, 211, 251))  # yellow
        bgr_image[self.cube == 6] = np.array((153, 51, 0))  # blue

        bgr_image = cv2.resize(bgr_image, (500, 500), interpolation=cv2.INTER_AREA)
        cv2.imshow("rubic_2Dmap", bgr_image)

    def save_rubik_2Dmap(self):
        try:
            os.mkdir(f"savefiles/{datetime.now():%Y%m%d}")
        except FileExistsError:
            pass

        with open(f"savefiles/{datetime.now():%Y%m%d}/cube_{datetime.now():%H%M%S}.pkl", "wb") as f:
            pickle.dump(self.cube, f)

        print("Successfully saved 2D Rubik map")


class RubikCubeCamera(Entity):
    def __init__(self, initial_position=(8, 8, -12), rotate_speed=130,
                 return_speed=10, save_path="", **kwargs):
        super().__init__(**kwargs)
        self.app = Ursina()
        self.target = Entity(model='cube', scale=3 / np.sqrt(2), color=color.black)
        self.rotator = Entity()
        self.rotate_speed = rotate_speed
        self.return_speed = return_speed
        self.cubes = []
        self.action_mode = False
        self.colors = {
            0: color.clear, 1: color.white, 2: color.orange, 3: color.green,
            4: color.red, 5: color.yellow, 6: color.azure
        }

        self.save_path = save_path
        self.rubik = rubik_3x3x3(save_path=self.save_path)
        self.rubik.show_rubik_2Dmap()
        self.draw_ursina_cube()

        self.pivot = Entity(position=self.target.position)
        camera.parent = self.pivot
        camera.position = initial_position
        camera.rotation_z = -5
        camera.look_at(self.pivot)
        self.default_rotation = self.pivot.rotation

        text = (
            "Controls:\n"
            "Right Click + Drag : Orbit Camera\n"
            "Release Click      : Reset Camera\n"
            "S Key              : Save 2D Map\n\n"
            "Cube Rotation (Numpad):\n"
            "-----------------------\n"
            "R : * |  R' : 3\n"
            "L : 2   |  L' : /\n"
            "U : 7   |  U' : -\n"
            "D : +   |  D' : 4\n"
            "F : 6   |  F' : 5\n"
            "B : 9   |  B' : 8"
        )
        Text(text=text, position=(-0.7, 0.45), origin=(-0.5, 0.5))

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
        # Threshold > 0.5 or < -0.5 works correctly
        for e in self.cubes:
            if side_name == '*' and e.world_x > 0.5:
                e.world_parent = self.rotator
            elif side_name == '3' and e.world_x > 0.5:
                e.world_parent = self.rotator
            elif side_name == '2' and e.world_x < -0.5:
                e.world_parent = self.rotator
            elif side_name == '/' and e.world_x < -0.5:
                e.world_parent = self.rotator
            elif side_name == '7' and e.world_y > 0.5:
                e.world_parent = self.rotator
            elif side_name == '-' and e.world_y > 0.5:
                e.world_parent = self.rotator
            elif side_name == '+' and e.world_y < -0.5:
                e.world_parent = self.rotator
            elif side_name == '4' and e.world_y < -0.5:
                e.world_parent = self.rotator
            elif side_name == '5' and e.world_z < -0.5:
                e.world_parent = self.rotator
            elif side_name == '6' and e.world_z < -0.5:
                e.world_parent = self.rotator
            elif side_name == '9' and e.world_z > 0.5:
                e.world_parent = self.rotator
            elif side_name == '8' and e.world_z > 0.5:
                e.world_parent = self.rotator

        # Animation axis
        if side_name in ['*', '3', '2', '/']:
            axis = 'rotation_x'
        elif side_name in ['7', '-', '+', '4']:
            axis = 'rotation_y'
        elif side_name in ['5', '9', '6', '8']:
            axis = 'rotation_z'

        # Apply logic
        mapping = {
            '*': self.rubik.R, '3': self.rubik.Ri,
            '2': self.rubik.L, '/': self.rubik.Li,
            '7': self.rubik.U, '-': self.rubik.Ui,
            '+': self.rubik.D, '4': self.rubik.Di,
            '6': self.rubik.F, '5': self.rubik.Fi,
            '8': self.rubik.B, '9': self.rubik.Bi
        }
        if side_name in mapping:
            mapping[side_name]()

        self.rubik.show_rubik_2Dmap()

        # Rotation Angle
        angle = 90 if side_name in ['*', '/', '7', '9', '4', '6'] else -90
        self.rotator.animate(axis, angle, duration=0.06)
        invoke(self.reset_structure, delay=0.08)

    def reset_structure(self):
        for e in self.cubes:
            e.world_parent = scene
        self.action_mode = False

    def input(self, key):
        k = key.upper()
        if k in ['*', '2', '/', '3', '7', '-', '+', '4', '8', '9', '6', '5']:
            self.rotate_side(k)
        elif k == "S":
            self.rubik.save_rubik_2Dmap()
        elif k == "R":
            self.reset_cube_state()
        elif k == "V":
            self.refresh_view()

    def draw_ursina_cube(self):
        cube_state = self.rubik.cube

        # 2x2x2 means 8 cubes. Positions are -1 and 0 and 1
        positions = [-1, 0, 1]

        for x in positions:
            for y in positions:
                for z in positions:
                    c = Entity(model='cube', color=color.black, position=(x, y, z), scale=0.99)
                    self.cubes.append(c)

        # Config adjusted for 2x2 indices
        # Grid layout: U[0:3, 3:6], L[3:6, 0:3], F[3:6, 3:6], R[3:6, 6:9], D[6:9, 3:6], B[6:9, 6:9]
        faces_config = [
            # Top (U)
            {'slice': (0, 3, 3, 6), 'pos': lambda r, c: Vec3(c - 1, 1, 1 - r), 'rot': (90, 0, 0)},
            # Left (L)
            {'slice': (3, 6, 0, 3), 'pos': lambda r, c: Vec3(-1, 1 - r, 1 - c), 'rot': (0, 90, 0)},
            # Front (F)
            {'slice': (3, 6, 3, 6), 'pos': lambda r, c: Vec3(c - 1, 1 - r, -1), 'rot': (0, 0, 0)},
            # Right (R)
            {'slice': (3, 6, 6, 9), 'pos': lambda r, c: Vec3(1, 1 - r, c - 1), 'rot': (0, -90, 0)},
            # Bottom (D)
            {'slice': (6, 9, 3, 6), 'pos': lambda r, c: Vec3(c - 1, -1, r - 1), 'rot': (-90, 0, 0)},
            # Back (B)
            {'slice': (6, 9, 6, 9), 'pos': lambda r, c: Vec3(c - 1, 1 - r, 1), 'rot': (0, 180, 0)},
        ]

        for config in faces_config:
            rs, re, cs, ce = config['slice']
            face_data = cube_state[rs:re, cs:ce]
            target_rot = Vec3(*config['rot'])

            for r in range(3):
                for c in range(3):
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

    def refresh_view(self):
        """現在のself.rubikの状態に基づいて描画をやり直す"""

        # 1. 既存のキューブEntityをシーンから削除
        for c in self.cubes:
            destroy(c)

        # 2. リストを空にする
        self.cubes.clear()

        # 3. 回転アニメーション用の親Entityの状態をリセット
        self.rotator.rotation = (0, 0, 0)
        self.action_mode = False

        # 4. 現在の内部データ(self.rubik.cube)に基づいて再描画
        self.draw_ursina_cube()

    # ---------------------------------------------------------
    # 【追加】完全に初期状態に戻すメソッド（論理リセット＋描画リセット）
    # ---------------------------------------------------------
    def reset_cube_state(self):
        """内部データと見た目の両方を初期化する"""
        print("Resetting Cube...")

        # 内部ロジッククラスを再インスタンス化（または初期化メソッドを呼ぶ）
        # ※引数は __init__ で受け取ったものと同じものを使う必要があります
        # ここでは初期化時のパラメータを保持していないため、簡易的に再作成します
        self.rubik = rubik_3x3x3(save_path=self.save_path)
        self.rubik.show_rubik_2Dmap()

        # オートソルブモードなどをリセット
        self.auto_solve_mode = False
        self.rubik.phase = 1

        # 見た目を更新
        self.refresh_view()

    def run_app(self):
        window.color = color.dark_gray
        window.title = '3x3 Rubik\'s Cube'
        self.app.run()


if __name__ == '__main__':
    rcc = RubikCubeCamera()
    rcc.run_app()
