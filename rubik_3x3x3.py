import os
import pickle
import random
from datetime import datetime

import cv2
import numpy as np
from ursina import EditorCamera, Entity, Ursina, Vec3, color, window


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


if __name__ == "__main__":
    # --- 設定 ---
    app = Ursina()
    window.title = 'Rubik\'s Cube Visualizer'
    window.color = color.gray  # 背景色

    # --- カメラ設定 ---
    EditorCamera()

    # --- 変数定義 ---
    # すべてのシール(Entity)を管理するリスト
    cubes = []

    # 回転アニメーション用のヘルパー（透明な回転軸）
    rotator = Entity()

    # アニメーション中かどうかを防ぐフラグ
    action_mode = False

    # 色の定義 (ユーザーの数字定義に合わせる)
    colors = {
        0: color.clear,      # 無効領域
        1: color.white,      # Top
        2: color.orange,     # Left
        3: color.green,      # Front
        4: color.red,        # Right
        5: color.yellow,     # Bottom
        6: color.azure       # Back (青系)
    }
