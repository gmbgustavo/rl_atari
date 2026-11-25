import argparse
import ctypes as C
from pathlib import Path

import numpy as np
from PIL import Image

from space_invaders_env import SpaceInvadersEnv
from train_ppo import treinar


class GameInfo(C.Structure):
    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


Environment = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
Video = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
Audio = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AudioBatch = C.CFUNCTYPE(C.c_size_t, C.POINTER(C.c_int16), C.c_size_t)
InputPoll = C.CFUNCTYPE(None)
InputState = C.CFUNCTYPE(C.c_int16, *([C.c_uint] * 4))

def frame_to_rgb(frame):
    raw, width, height, pitch, fmt = frame
    dtype = np.dtype(np.uint32 if fmt == 1 else np.uint16)
    pixels = np.ndarray(
        (height, width),
        dtype=dtype,
        buffer=raw,
        strides=(pitch, dtype.itemsize),
    ).astype(np.uint32)

    if fmt == 1:  # XRGB8888
        rgb = [(pixels >> shift) & 255 for shift in (16, 8, 0)]
    else:  # RGB565 ou 0RGB1555
        red_shift, green_max = (11, 63) if fmt == 2 else (10, 31)
        rgb = [
            ((pixels >> red_shift) & 31) * 255 // 31,
            ((pixels >> 5) & green_max) * 255 // green_max,
            (pixels & 31) * 255 // 31,
        ]

    return np.stack(rgb, axis=-1).astype(np.uint8)

def capture(core_path, rom_path):
    core = C.CDLL(str(core_path.resolve()))
    pixel_format = 0  # Padrão Libretro: 0RGB1555
    last_frame = None

    @Environment
    def environment(command, data):
        nonlocal pixel_format
        if command == 10:  # RETRO_ENVIRONMENT_SET_PIXEL_FORMAT
            fmt = C.cast(data, C.POINTER(C.c_int))[0]
            if fmt in (0, 1, 2):  # 0RGB1555, XRGB8888, RGB565
                pixel_format = fmt
                return True
        return False

    @Video
    def video(data, width, height, pitch):
        nonlocal last_frame
        if data and width and height:
            # O buffer pertence ao core: copie antes de sair do callback.
            last_frame = (
                C.string_at(data, pitch * height),
                width, height, pitch, pixel_format,
            )

    buttons = set()
    axis_x = 0
    reads = 0

    @InputState
    def input_state(port, device, index, button):
        nonlocal reads

        if port != 0 or index != 0:
            return 0

        if device == 1:  # Joypad
            if button == 256:  # Libretro também pode pedir a máscara inteira
                value = sum(1 << b for b in buttons)
            else:
                value = int(button in buttons)
        elif device == 5 and button == 0:  # Analógico esquerdo, eixo X
            value = axis_x
        else:
            value = 0

        reads += int(value != 0)
        return value

    callbacks = {
        "environment": environment,
        "video_refresh": video,
        "audio_sample": Audio(lambda left, right: None),
        "audio_sample_batch": AudioBatch(lambda data, frames: frames),
        "input_poll": InputPoll(lambda: None),
        "input_state": input_state,
    }
    for name, callback in callbacks.items():
        setter = getattr(core, f"retro_set_{name}")
        setter.argtypes = [type(callback)]
        setter.restype = None
        setter(callback)

    for name in ("retro_init", "retro_run", "retro_unload_game", "retro_deinit"):
        fn = getattr(core, name)
        fn.argtypes = []
        fn.restype = None

    core.retro_load_game.argtypes = [C.POINTER(GameInfo)]
    core.retro_load_game.restype = C.c_bool

    # Mantenha os bytes da ROM vivos até retro_unload_game().
    rom = rom_path.read_bytes()
    buffer = C.create_string_buffer(rom)
    game = GameInfo(
        str(rom_path.resolve()).encode("utf-8"),
        C.cast(buffer, C.c_void_p),
        len(rom),
        None,
    )

    core.retro_init()
    try:
        if not core.retro_load_game(C.byref(game)):
            raise RuntimeError("Stella recusou a ROM. Use o arquivo .a26/.bin descompactado.")
        snapshots = {}
        try:
            for _ in range(120):
                core.retro_run()

            # START no Libretro aciona o RESET do console.
            buttons.clear()
            axis_x = 0
            buttons.add(3)
            for _ in range(8):
                core.retro_run()

            buttons.clear()
            for _ in range(150):
                core.retro_run()

            # RETRO_MEMORY_SYSTEM_RAM = 2
            core.retro_get_memory_data.argtypes = [C.c_uint]
            core.retro_get_memory_data.restype = C.c_void_p
            core.retro_get_memory_size.argtypes = [C.c_uint]
            core.retro_get_memory_size.restype = C.c_size_t

            ram_ptr = core.retro_get_memory_data(2)
            ram_size = core.retro_get_memory_size(2)
            if not ram_ptr or ram_size != 128:
                raise RuntimeError(f"Esperava 128 bytes de RAM do Atari; recebi {ram_size}.")

            def read_stats():
                ram = C.string_at(ram_ptr, ram_size)

                def bcd(value):
                    return 10 * (value >> 4) + (value & 0x0F)

                score = 100 * bcd(ram[0x66]) + bcd(ram[0x68])
                lives = ram[0x49]
                game_over = bool(ram[0x18] & 0x80) or lives == 0
                return score, lives, game_over

            print("Após reset (placar, vidas, fim):", read_stats())

            snapshots["INICIO"] = last_frame

            # ID da ação -> botões pressionados simultaneamente
            actions = (
                (),  # 0: neutro
                (6,),  # 1: esquerda
                (7,),  # 2: direita
                (0,),  # 3: tiro
                (6, 0),  # 4: esquerda + tiro
                (7, 0),  # 5: direita + tiro
            )

            def step(action):
                buttons.clear()
                buttons.update(actions[action])
                for _ in range(40):
                    core.retro_run()
                return last_frame

            previous = read_stats()

            def reset_game():
                buttons.clear()
                buttons.add(3)  # RESET do console
                for _ in range(8):
                    core.retro_run()

                buttons.clear()
                for _ in range(160):
                    core.retro_run()

                return last_frame

            env = SpaceInvadersEnv(reset_game, step, read_stats, frame_to_rgb)

            for episode in range(1, 3):
                obs, info = env.reset()
                env = SpaceInvadersEnv(reset_game, step, read_stats, frame_to_rgb)
                treinar(env)

            snapshots["FIM_TESTE"] = last_frame
            print("Estado final:", read_stats())

        finally:
            core.retro_unload_game()
    finally:
        core.retro_deinit()
        for name, frame in snapshots.items():
            if frame is None:
                print(f"{name}: nenhum frame recebido")
                continue

            raw, width, height, pitch, fmt = frame
            dtype = np.uint32 if fmt == 1 else np.uint16
            pixels = np.ndarray(
                (height, width), dtype=dtype, buffer=raw,
                strides=(pitch, np.dtype(dtype).itemsize),
            ).astype(np.uint32)

            if fmt == 1:  # XRGB8888
                rgb = [(pixels >> shift) & 255 for shift in (16, 8, 0)]
            else:  # RGB565 ou 0RGB1555
                red_shift, green_max = (11, 63) if fmt == 2 else (10, 31)
                rgb = [
                    ((pixels >> red_shift) & 31) * 255 // 31,
                    ((pixels >> 5) & green_max) * 255 // green_max,
                    (pixels & 31) * 255 // 31,
                ]

            filename = f"{name}.png"
            Image.fromarray(frame_to_rgb(frame)).save(f"{name}.png")
            print(f"OK: {filename} salvo ({width}×{height})")

    if last_frame is None:
        raise RuntimeError("O core não entregou nenhum frame.")

    raw, width, height, pitch, fmt = last_frame
    dtype = np.uint32 if fmt == 1 else np.uint16
    pixels = np.ndarray(
        (height, width), dtype=dtype, buffer=raw,
        strides=(pitch, np.dtype(dtype).itemsize),
    ).astype(np.uint32)

    if fmt == 1:  # XRGB8888
        rgb = [(pixels >> shift) & 255 for shift in (16, 8, 0)]
    else:
        red_shift, green_max = (11, 63) if fmt == 2 else (10, 31)
        rgb = [
            ((pixels >> red_shift) & 31) * 255 // 31,
            ((pixels >> 5) & green_max) * 255 // green_max,
            (pixels & 31) * 255 // 31,
        ]

    Image.fromarray(np.stack(rgb, axis=-1).astype(np.uint8)).save("frame.png")
    print(f"OK: frame.png salvo ({width}×{height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=Path)
    parser.add_argument("rom", type=Path)
    args = parser.parse_args()
    capture(args.core, args.rom)