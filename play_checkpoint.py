import argparse
import ctypes as C
import os
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from collections import deque


# Joypad Libretro: B=0, START=3, LEFT=6, RIGHT=7.
FIRE, START, LEFT, RIGHT = 1 << 0, 1 << 3, 1 << 6, 1 << 7
ACTIONS = (0, LEFT, RIGHT, FIRE, LEFT | FIRE, RIGHT | FIRE)
SYSTEM_RAM = 2

Environment = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VideoRefresh = C.CFUNCTYPE(
    None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t
)
AudioSample = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AudioBatch = C.CFUNCTYPE(C.c_size_t, C.POINTER(C.c_int16), C.c_size_t)
InputPoll = C.CFUNCTYPE(None)
InputState = C.CFUNCTYPE(
    C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint
)


class GameInfo(C.Structure):
    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


class SystemInfo(C.Structure):
    _fields_ = [
        ("library_name", C.c_char_p),
        ("library_version", C.c_char_p),
        ("valid_extensions", C.c_char_p),
        ("need_fullpath", C.c_bool),
        ("block_extract", C.c_bool),
    ]


class Stella:
    def __init__(self, core_path):
        self.core = C.CDLL(str(core_path.resolve()))
        self.buttons = 0
        self.pixel_format = 0  # Formato padrão do Libretro: 0RGB1555.
        self.frame = None
        self.loaded = False

        signatures = {
            "retro_set_environment": ([Environment], None),
            "retro_set_video_refresh": ([VideoRefresh], None),
            "retro_set_audio_sample": ([AudioSample], None),
            "retro_set_audio_sample_batch": ([AudioBatch], None),
            "retro_set_input_poll": ([InputPoll], None),
            "retro_set_input_state": ([InputState], None),
            "retro_get_system_info": ([C.POINTER(SystemInfo)], None),
            "retro_init": ([], None),
            "retro_deinit": ([], None),
            "retro_load_game": ([C.POINTER(GameInfo)], C.c_bool),
            "retro_unload_game": ([], None),
            "retro_run": ([], None),
            "retro_get_memory_data": ([C.c_uint], C.c_void_p),
            "retro_get_memory_size": ([C.c_uint], C.c_size_t),
        }
        for name, (args, result) in signatures.items():
            func = getattr(self.core, name)
            func.argtypes = args
            func.restype = result

        # Mantenha referências aos callbacks enquanto o core estiver ativo.
        self.callbacks = (
            ("retro_set_environment", Environment(self.environment)),
            ("retro_set_video_refresh", VideoRefresh(self.video)),
            ("retro_set_audio_sample", AudioSample(lambda left, right: None)),
            ("retro_set_audio_sample_batch", AudioBatch(lambda data, n: n)),
            ("retro_set_input_poll", InputPoll(lambda: None)),
            ("retro_set_input_state", InputState(self.input_state)),
        )
        for setter, callback in self.callbacks:
            getattr(self.core, setter)(callback)

    def __enter__(self):
        self.core.retro_init()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.loaded:
                self.core.retro_unload_game()
        finally:
            self.core.retro_deinit()

    def environment(self, cmd, data):
        if cmd == 10 and data:  # RETRO_ENVIRONMENT_SET_PIXEL_FORMAT
            fmt = C.cast(data, C.POINTER(C.c_int))[0]
            if fmt in (0, 1, 2):
                self.pixel_format = fmt
                return True
        elif cmd == 3 and data:  # RETRO_ENVIRONMENT_GET_CAN_DUPE
            C.cast(data, C.POINTER(C.c_bool))[0] = True
            return True
        elif cmd == (51 | 0x10000):  # GET_INPUT_BITMASKS
            return True
        return False

    def video(self, data, width, height, pitch):
        if data:
            self.frame = (
                C.string_at(data, pitch * height),
                width, height, pitch, self.pixel_format,
            )

    def input_state(self, port, device, index, button):
        if port != 0 or index != 0 or device != 1:
            return 0
        if button == 256:
            return self.buttons
        return int(button < 16 and bool(self.buttons & (1 << button)))

    def load_rom(self, rom_path):
        path = rom_path.resolve()
        info = SystemInfo()
        self.core.retro_get_system_info(C.byref(info))

        self.rom_buffer = None
        data, size = None, 0
        if not info.need_fullpath:
            raw = path.read_bytes()
            size = len(raw)
            self.rom_buffer = C.create_string_buffer(raw)
            data = C.cast(self.rom_buffer, C.c_void_p)

        self.game_info = GameInfo(os.fsencode(path), data, size, None)
        self.loaded = bool(self.core.retro_load_game(C.byref(self.game_info)))
        if not self.loaded:
            raise RuntimeError(f"O Stella não conseguiu carregar a ROM: {path}")

        self.run_frames(120)

    def run_frames(self, count, buttons=0):
        self.buttons = buttons
        for _ in range(count):
            self.core.retro_run()

    def reset(self):
        self.frame = None
        self.run_frames(8, START)
        self.run_frames(160)

    def stats(self):
        ptr = self.core.retro_get_memory_data(SYSTEM_RAM)
        size = self.core.retro_get_memory_size(SYSTEM_RAM)
        if not ptr or size < 128:
            raise RuntimeError(f"RAM do Atari indisponível: {size} bytes")

        ram = C.string_at(ptr, 128)

        def bcd(value):
            return 10 * (value >> 4) + (value & 15)

        score = 100 * bcd(ram[0x66]) + bcd(ram[0x68])
        lives = ram[0x49]
        game_over = bool(ram[0x18] & 0x80) or lives == 0
        return score, lives, game_over

    def observation(self):
        if self.frame is None:
            raise RuntimeError("O Stella ainda não entregou um frame.")

        raw, width, height, pitch, fmt = self.frame
        dtype = np.dtype(np.uint32 if fmt == 1 else np.uint16)
        pixels = np.ndarray(
            (height, width),
            dtype=dtype,
            buffer=raw,
            strides=(pitch, dtype.itemsize),
        ).astype(np.uint32)

        if fmt == 1:  # XRGB8888
            channels = [(pixels >> shift) & 255 for shift in (16, 8, 0)]
        else:  # RGB565 ou 0RGB1555
            red_shift, green_max = (11, 63) if fmt == 2 else (10, 31)
            channels = [
                ((pixels >> red_shift) & 31) * 255 // 31,
                ((pixels >> 5) & green_max) * 255 // green_max,
                (pixels & 31) * 255 // 31,
            ]

        rgb = np.stack(channels, axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb).convert("L")
        image = image.resize((84, 84), Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.uint8, copy=True)[..., None]


def main():
    parser = argparse.ArgumentParser(
        description="Joga Space Invaders com um checkpoint PPO, sem treinar."
    )
    parser.add_argument("core", type=Path, help="Caminho para stella_libretro.dll")
    parser.add_argument("rom", type=Path, help="Caminho para a ROM .a26")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("ppo_space_invaders.zip")
    )
    parser.add_argument("--partidas", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amostrar", action="store_true",
        help="Amostra ações como na avaliação do treino, em vez de usar a mais provável."
    )
    args = parser.parse_args()

    if args.partidas < 1:
        parser.error("--partidas deve ser pelo menos 1")
    for path in (args.core, args.rom, args.checkpoint):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")

    model = PPO.load(str(args.checkpoint), device=args.device)
    print(f"Checkpoint: {args.checkpoint} | device: {model.device}")

    with Stella(args.core) as stella:
        stella.load_rom(args.rom)

        for episode in range(1, args.partidas + 1):
            stella.reset()
            score, lives, game_over = stella.stats()
            if (score, lives, game_over) != (0, 3, False):
                raise RuntimeError(f"Reset inesperado: {(score, lives, game_over)}")

            frames = deque(maxlen=4)
            first_frame = stella.observation()
            for _ in range(4):
                frames.append(first_frame.copy())
            obs = np.concatenate(tuple(frames), axis=-1).copy()
            total = 0

            for steps in range(1, 1801):
                action, _ = model.predict(obs, deterministic=not args.amostrar)
                stella.run_frames(4, ACTIONS[int(np.asarray(action).item())])

                new_score, lives, game_over = stella.stats()
                total += (new_score - score) % 10_000
                score = new_score

                if game_over or steps == 1800:
                    break
                frames.append(stella.observation().copy())
                obs = np.concatenate(tuple(frames), axis=-1).copy()

            reason = "game over" if game_over else "limite de passos"
            print(
                f"Partida {episode}: {reason} | placar={score} | "
                f"vidas={lives} | recompensa={total} | passos={steps}"
            )


if __name__ == "__main__":
    main()