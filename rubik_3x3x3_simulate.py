import numpy as np
from ursina import EditorCamera, Entity, Ursina, Vec3, color, distance, invoke, scene, window

from rubik_3x3x3 import rubik_3x3x3

app = Ursina()
window.title = 'Rubik\'s Cube Playable'
window.color = color.gray

# --- データ定義 ---
cube_data = rubik_3x3x3()
cube_data.show_rubik_2Dmap()

# --- カメラ設定 ---
ec = EditorCamera()

# --- 変数定義 ---
# すべてのシール(Entity)を管理するリスト
cubes = []

# 回転アニメーション用のヘルパー（透明な回転軸）
rotator = Entity()

# アニメーション中かどうかを防ぐフラグ
action_mode = False


def rotate_side(side_name):
    global action_mode
    global cube_data
    if action_mode:
        return  # アニメーション中は操作を受け付けない
    action_mode = True

    # 1. 回転軸をリセット
    rotator.rotation = (0, 0, 0)

    # 2. 回転対象のグループ化
    # 現在の位置に基づいて、回転させたい面のパーツだけをrotatorの子にする
    for e in cubes:
        # ワールド座標での位置を使って判定
        if side_name == '*' and e.world_x > 0.5:
            e.world_parent = rotator
        elif side_name == '3' and e.world_x > 0.5:
            e.world_parent = rotator
        elif side_name == '2' and e.world_x < -0.5:
            e.world_parent = rotator
        elif side_name == '/' and e.world_x < -0.5:
            e.world_parent = rotator
        elif side_name == '7' and e.world_y > 0.5:
            e.world_parent = rotator
        elif side_name == '-' and e.world_y > 0.5:
            e.world_parent = rotator
        elif side_name == '+' and e.world_y < -0.5:
            e.world_parent = rotator
        elif side_name == '4' and e.world_y < -0.5:
            e.world_parent = rotator
        elif side_name == '8' and e.world_z == 0:
            e.world_parent = rotator  # 手前
        elif side_name == '9' and e.world_z == 0:
            e.world_parent = rotator  # 手前
        elif side_name == '6' and e.world_z > 0.5:
            e.world_parent = rotator  # 奥
        elif side_name == '5' and e.world_z > 0.5:
            e.world_parent = rotator  # 奥

    # 3. アニメーション実行 (0.5秒で90度)
    # evalを使うと文字列から回転方向を指定できます（例: rotation_x）
    if side_name in ['*', '3', '2', '/']:
        axis = 'rotation_x'
    elif side_name in ['7', '-', '+', '4']:
        axis = 'rotation_y'
    elif side_name in ['8', '6', '9', '5']:
        axis = 'rotation_z'

    if side_name == "*":
        cube_data.col_r_up()
    elif side_name == "3":
        cube_data.col_r_down()
    elif side_name == "2":
        cube_data.col_l_down()
    elif side_name == "/":
        cube_data.col_l_up()
    elif side_name == "7":
        cube_data.row_top_left()
    elif side_name == "-":
        cube_data.row_top_right()
    elif side_name == "+":
        cube_data.row_btm_right()
    elif side_name == "4":
        cube_data.row_btm_right()
    elif side_name == "8":
        cube_data.layer2_ccw()
    elif side_name == "9":
        cube_data.layer2_cw()
    elif side_name == "6":
        cube_data.layer3_cw()
    elif side_name == "5":
        cube_data.layer3_ccw()
    elif side_name == "s":
        cube_data.save_rubik_2Dmap()

    cube_data.show_rubik_2Dmap()

    # 回転方向の正負（時計回り・反時計回り）
    angle = 90 if side_name in ['*', '7', '6', '4', '9'] else -90

    rotator.animate(axis, angle, duration=0.06)

    # 4. アニメーション終了後の処理
    invoke(reset_structure, delay=0.08)


def reset_structure():
    global action_mode
    # 親子関係を解除し、回転後の座標を確定させる
    for e in cubes:
        e.world_parent = scene
    action_mode = False


# --- キー入力イベント ---
def input(key):
    # Shiftキーなどを押しながらの場合は小文字になるため upper() で統一
    k = key.upper()

    if k in ['*', '2', '/', '3', '7', '-', '+', '4', '8', '6', '9', '5', 's']:
        rotate_side(k)


# --- 描画ロジック ---
colors = {
    0: color.clear, 1: color.white, 2: color.orange, 3: color.green,
    4: color.red, 5: color.yellow, 6: color.azure
}


def draw_ursina_cube(cube_state):
    # シールだけでなく、黒い土台も含めて管理するため、少し構造を変えます
    # 3x3x3 の 27個の「小さな黒いキューブ」を作り、そこにシールを貼る形にします

    # 27個のキューブ生成
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                # 黒い小さなキューブ（これが物理的な実体になります）
                c = Entity(model='cube', color=color.black, position=(x, y, z), scale=0.99)
                cubes.append(c)

    # 黒いコア
    Entity(model='cube', scale=3 / np.sqrt(2), color=color.black)

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
                for cube_entity in cubes:
                    if distance(cube_entity.position, target_pos) < 0.1:
                        # 黒キューブが見つかったら、その表面に色付きの板を追加
                        sticker = Entity(
                            parent=cube_entity,
                            model='quad',
                            color=colors[color_code],
                            scale=0.9,
                            texture='white_cube'
                        )

                        sticker.world_position = target_pos  # 既に計算済みの正しい位置
                        sticker.world_rotation = target_rot

                        sticker.world_position += sticker.back * 0.6

                        break


draw_ursina_cube(cube_data.cube)
app.run()
