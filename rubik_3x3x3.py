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
    def __init__(self, save_path=""):
        self.save_path = ""
        if save_path:
            with open(save_path, "rb") as f:
                self.cube = pickle.load(f)
        else:
            self.cube = np.array([
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [2, 2, 2, 3, 3, 3, 4, 4, 4],
                [2, 2, 2, 3, 3, 3, 4, 4, 4],
                [2, 2, 2, 3, 3, 3, 4, 4, 4],
                [0, 0, 0, 5, 5, 5, 6, 6, 6],
                [0, 0, 0, 5, 5, 5, 6, 6, 6],
                [0, 0, 0, 5, 5, 5, 6, 6, 6]
            ], dtype=np.uint8)
            self.shuffle()

    def shuffle(self):  # noqa: C901
        shuffle_num = 50  # シャッフルする回数（20~50回がいいらしい）
        manipulate_num = 12  # 操作の種類の数
        for i in range(shuffle_num):
            rand = int(random.random() * manipulate_num)
            if rand == 0:
                self.col_r_up()
            elif rand == 1:
                self.col_r_down()
            elif rand == 2:
                self.col_l_up()
            elif rand == 3:
                self.col_l_down()
            elif rand == 4:
                self.row_top_right()
            elif rand == 5:
                self.row_top_left()
            elif rand == 6:
                self.row_btm_right()
            elif rand == 7:
                self.row_btm_left()
            elif rand == 8:
                self.layer2_cw()
            elif rand == 9:
                self.layer2_ccw()
            elif rand == 10:
                self.layer3_cw()
            elif rand == 11:
                self.layer3_ccw()

    def col_r_up(self):
        y_indices = np.array([3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2])
        x_indices = np.array([5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5])
        y_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        x_prime_indices = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[3:6, 6:9] = np.rot90(self.cube[3:6, 6:9], k=-1)

    def col_r_down(self):
        y_indices = np.array([8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        x_indices = np.array([8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        y_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        x_prime_indices = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[3:6, 6:9] = np.rot90(self.cube[3:6, 6:9], k=1)

    def col_l_up(self):
        y_indices = np.array([3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2])
        x_indices = np.array([3, 3, 3, 3, 3, 3, 6, 6, 6, 3, 3, 3])
        y_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        x_prime_indices = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[3:6, 0:3] = np.rot90(self.cube[3:6, 0:3], k=1)

    def col_l_down(self):
        y_indices = np.array([8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        x_indices = np.array([6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        y_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        x_prime_indices = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[3:6, 0:3] = np.rot90(self.cube[3:6, 0:3], k=-1)

    def row_top_right(self):
        y_indices = np.array([6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        x_indices = np.array([8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        y_prime_indices = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6])
        x_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[0:3, 3:6] = np.rot90(self.cube[0:3, 3:6], k=1)

    def row_top_left(self):
        y_indices = np.array([3, 3, 3, 3, 3, 3, 6, 6, 6, 3, 3, 3])
        x_indices = np.array([3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2])
        y_prime_indices = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6])
        x_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[0:3, 3:6] = np.rot90(self.cube[0:3, 3:6], k=-1)

    def row_btm_right(self):
        y_indices = np.array([8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        x_indices = np.array([8, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        y_prime_indices = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8])
        x_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[6:9, 3:6] = np.rot90(self.cube[6:9, 3:6], k=-1)

    def row_btm_left(self):
        y_indices = np.array([5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5])
        x_indices = np.array([3, 4, 5, 6, 7, 8, 8, 7, 6, 0, 1, 2])
        y_prime_indices = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8])
        x_prime_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[6:9, 3:6] = np.rot90(self.cube[6:9, 3:6], k=1)

    # cw: crockwise, ccw counter-clockwise
    def layer2_cw(self):
        y_indices = np.array([5, 4, 3, 1, 1, 1, 3, 4, 5, 7, 7, 7])
        x_indices = np.array([1, 1, 1, 3, 4, 5, 7, 7, 7, 5, 4, 3])
        y_prime_indices = np.array([1, 1, 1, 3, 4, 5, 7, 7, 7, 5, 4, 3])
        x_prime_indices = np.array([3, 4, 5, 7, 7, 7, 5, 4, 3, 1, 1, 1])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]

    def layer2_ccw(self):
        y_indices = np.array([3, 4, 5, 7, 7, 7, 5, 4, 3, 1, 1, 1])
        x_indices = np.array([7, 7, 7, 5, 4, 3, 1, 1, 1, 3, 4, 5])
        y_prime_indices = np.array([1, 1, 1, 3, 4, 5, 7, 7, 7, 5, 4, 3])
        x_prime_indices = np.array([3, 4, 5, 7, 7, 7, 5, 4, 3, 1, 1, 1])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]

    def layer3_cw(self):
        y_indices = np.array([5, 4, 3, 0, 0, 0, 3, 4, 5, 8, 8, 8])
        x_indices = np.array([0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3])
        y_prime_indices = np.array([0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3])
        x_prime_indices = np.array([3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[6:9, 6:9] = np.rot90(self.cube[6:9, 6:9], k=-1)

    def layer3_ccw(self):
        y_indices = np.array([3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0])
        x_indices = np.array([8, 8, 8, 5, 4, 3, 0, 0, 0, 3, 4, 5])
        y_prime_indices = np.array([0, 0, 0, 3, 4, 5, 8, 8, 8, 5, 4, 3])
        x_prime_indices = np.array([3, 4, 5, 8, 8, 8, 5, 4, 3, 0, 0, 0])
        self.cube[y_prime_indices, x_prime_indices] = self.cube[y_indices, x_indices]
        self.cube[6:9, 6:9] = np.rot90(self.cube[6:9, 6:9], k=1)

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

        with open(f"savefiles/{datetime.now():%Y%m%d}/{datetime.now():%H%M%S}.pkl", "wb") as f:
            pickle.dump(self.cube, f)

        print("Successfully saved 2D Rubik map")


class RubikCubeCamera(Entity):
    def __init__(self, initial_position=(8, 8, -12), rotate_speed=130, return_speed=10, **kwargs):
        super().__init__(**kwargs)
        self.app = Ursina()
        self.target = Entity(model='cube', scale=3 / np.sqrt(2), color=color.black)
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

        self.cube_data = rubik_3x3x3()
        self.cube_data.show_rubik_2Dmap()
        self.draw_ursina_cube()

        # 1. ピボット（回転の中心軸）をターゲットの位置に作成
        # これが回転することでカメラが周囲を回る
        self.pivot = Entity(position=self.target.position)

        # 2. メインカメラの設定
        # カメラをピボットの子要素にする（ピボットが動けばカメラもついていく）
        camera.parent = self.pivot
        camera.position = self.initial_position
        camera.rotation_z = -5  # マイナスを指定すると右（時計回り）に傾く

        # 3. 常にターゲット（ピボット）を向くように設定
        # 親子関係があるため、一度設定すればピボットが回転しても向きは維持される
        camera.look_at(self.pivot)

        # 4. 初期回転（リセット時の戻り先）を保存。通常は(0,0,0)
        self.default_rotation = self.pivot.rotation

        # 説明テキスト
        Text(text='Hold Right Mouse to Orbit\nRelease to Reset', position=(-0.7, 0.45), origin=(-0.5, 0.5))

    def update(self):
        # 条件：右クリック押下中は動き、離したら初期位置に戻る
        if held_keys['right mouse']:
            # マウスの移動量(velocity)に応じてピボットを回転させる
            # rotation_y: 横回転 (マウスX移動)
            # rotation_x: 縦回転 (マウスY移動) - 操作感を自然にするためマイナスを入れる場合が多い
            self.pivot.rotation_y += mouse.velocity[0] * self.rotate_speed
            self.pivot.rotation_x -= mouse.velocity[1] * self.rotate_speed * 2
        else:
            # 右クリックされていない時は、初期回転(0,0,0)へ滑らかに戻る
            # lerp(現在値, 目標値, 速度 * 時間) を使用してスムーズな復帰を実現
            self.pivot.rotation = lerp(self.pivot.rotation, self.default_rotation, time.dt * self.return_speed)

    def rotate_side(self, side_name):  # noqa: C901
        if self.action_mode:
            return  # アニメーション中は操作を受け付けない
        self.action_mode = True

        # 1. 回転軸をリセット
        self.rotator.rotation = (0, 0, 0)

        # 2. 回転対象のグループ化
        # 現在の位置に基づいて、回転させたい面のパーツだけをrotatorの子にする
        for e in self.cubes:
            # ワールド座標での位置を使って判定
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
            elif side_name == '5' and e.world_z == 0:
                e.world_parent = self.rotator
            elif side_name == '6' and e.world_z == 0:
                e.world_parent = self.rotator
            elif side_name == '9' and e.world_z > 0.5:
                e.world_parent = self.rotator
            elif side_name == '8' and e.world_z > 0.5:
                e.world_parent = self.rotator

        # 3. アニメーション実行
        if side_name in ['*', '3', '2', '/']:
            axis = 'rotation_x'
        elif side_name in ['7', '-', '+', '4']:
            axis = 'rotation_y'
        elif side_name in ['5', '9', '6', '8']:
            axis = 'rotation_z'

        if side_name == "*":
            self.cube_data.col_r_up()
        elif side_name == "3":
            self.cube_data.col_r_down()
        elif side_name == "2":
            self.cube_data.col_l_down()
        elif side_name == "/":
            self.cube_data.col_l_up()
        elif side_name == "7":
            self.cube_data.row_top_left()
        elif side_name == "-":
            self.cube_data.row_top_right()
        elif side_name == "+":
            self.cube_data.row_btm_right()
        elif side_name == "4":
            self.cube_data.row_btm_right()
        elif side_name == "5":
            self.cube_data.layer2_ccw()
        elif side_name == "6":
            self.cube_data.layer2_cw()
        elif side_name == "9":
            self.cube_data.layer3_cw()
        elif side_name == "8":
            self.cube_data.layer3_ccw()

        self.cube_data.show_rubik_2Dmap()

        # 回転方向の正負（時計回り・反時計回り）
        angle = 90 if side_name in ['*', '/', '7', '9', '4', '6'] else -90

        self.rotator.animate(axis, angle, duration=0.06)

        # 4. アニメーション終了後の処理
        invoke(self.reset_structure, delay=0.08)

    def reset_structure(self):
        # 親子関係を解除し、回転後の座標を確定させる
        for e in self.cubes:
            e.world_parent = scene
        self.action_mode = False

    # --- キー入力イベント ---
    def input(self, key):
        k = key.upper()

        if k in ['*', '2', '/', '3', '7', '-', '+', '4', '8', '9', '6', '5']:
            self.rotate_side(k)
        elif k == "S":
            self.cube_data.save_rubik_2Dmap()

    def draw_ursina_cube(self):
        cube_state = self.cube_data.cube
        # 3x3x3 の 27個の「小さな黒いキューブ」を作り、そこにシールを貼る

        # 27個のキューブ生成
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    c = Entity(model='cube', color=color.black, position=(x, y, z), scale=0.99)
                    self.cubes.append(c)

        # 展開図データからシールを貼り付け
        # ここでは既存のロジックを使って「位置」を特定し、
        # その位置に最も近いキューブを見つけて色を塗ります

        faces_config = [
            # Top (上面): X軸で90度回して上に向ける
            {'slice': (0, 3, 3, 6), 'pos': lambda r, c: Vec3(c - 1, 1, 1 - r), 'rot': (90, 0, 0)},
            # Left (左面): Y軸で-90度回して左に向ける
            {'slice': (3, 6, 0, 3), 'pos': lambda r, c: Vec3(-1, 1 - r, 1 - c), 'rot': (0, 90, 0)},
            # Front (正面): 回転なし（そのまま手前を向く）
            {'slice': (3, 6, 3, 6), 'pos': lambda r, c: Vec3(c - 1, 1 - r, -1), 'rot': (0, 0, 0)},
            # Right (右面): Y軸で90度回して右に向ける
            {'slice': (3, 6, 6, 9), 'pos': lambda r, c: Vec3(1, 1 - r, c - 1), 'rot': (0, -90, 0)},
            # Bottom (底面): X軸で-90度回して下に向ける
            {'slice': (6, 9, 3, 6), 'pos': lambda r, c: Vec3(c - 1, -1, r - 1), 'rot': (-90, 0, 0)},
            # Back (背面): Y軸で180度回して奥に向ける
            {'slice': (6, 9, 6, 9), 'pos': lambda r, c: Vec3(c - 1, 1 - r, 1), 'rot': (0, 180, 0)},
        ]

        for config in faces_config:
            rs, re, cs, ce = config['slice']
            face_data = cube_state[rs:re, cs:ce]

            # 回転角度をVec3に変換
            target_rot = Vec3(*config['rot'])

            for r in range(3):
                for c in range(3):
                    color_code = face_data[r, c]
                    if color_code == 0:
                        continue

                    # このシールが貼られるべき座標
                    target_pos = config['pos'](r, c)

                    # その座標にある黒キューブを探す
                    # (浮動小数点の誤差を考慮して距離で判定)
                    for cube_entity in self.cubes:
                        if distance(cube_entity.position, target_pos) < 0.1:
                            # 黒キューブが見つかったら、その表面に色付きの板を追加
                            sticker = Entity(
                                parent=cube_entity,
                                model='quad',
                                color=self.colors[color_code],
                                scale=0.9,
                                texture='white_cube'
                            )

                            sticker.world_position = target_pos  # 既に計算済みの正しい位置
                            sticker.world_rotation = target_rot

                            sticker.world_position += sticker.back * 0.6

                            break

    def run_app(self):
        window.color = color.dark_gray
        window.title = 'Rubik\'s Cube Playable'
        self.app.run()


if __name__ == '__main__':
    rcc = RubikCubeCamera()
    rcc.run_app()
