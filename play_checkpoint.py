import argparse
from pathlib import Path
from statistics import mean, median

import torch
from stable_baselines3 import PPO

from space_invaders_env import PRIMEIRA_SEMENTE_AVALIACAO, criar_ambiente, jogar_partida
from train_ppo import SAIDA_PADRAO, politica_ppo


def main():
    parser = argparse.ArgumentParser(
        description="Joga Space Invaders com um checkpoint PPO, sem treinar."
    )
    parser.add_argument("core", type=Path, help="Caminho para o core Stella (.so ou .dll)")
    parser.add_argument("rom", type=Path, help="Caminho para a ROM de Space Invaders")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partidas", type=int, default=10)
    parser.add_argument(
        "--semente", type=int, default=PRIMEIRA_SEMENTE_AVALIACAO,
        help="Primeira semente das partidas",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--amostrar", action="store_true", help="Amostra ações como na avaliação do treino")
    modo.add_argument("--deterministico", action="store_true", help="Usa sempre a ação mais provável")
    parser.add_argument("--render", action="store_true", help="Abre uma janela para acompanhar o jogo")
    args = parser.parse_args()

    if args.checkpoint is None:
        best = SAIDA_PADRAO / "best.zip"
        anterior = Path("ppo_space_invaders.zip")
        legado = Path("checkpoints/ppo_sinvaders_4frames_latest.zip")
        args.checkpoint = next((p for p in (best, anterior) if p.is_file()), legado)
    if args.partidas < 1:
        parser.error("--partidas deve ser pelo menos 1")
    if args.semente < 0:
        parser.error("--semente deve ser não negativa")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponível; use --device auto ou cpu")
    for path in (args.core, args.rom, args.checkpoint):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")

    torch.set_num_threads(2)
    model = PPO.load(args.checkpoint, device=args.device)
    legado = not getattr(model, "space_invaders_v2", False)
    deterministica = args.deterministico or (legado and not args.amostrar)
    modo = "ação mais provável" if deterministica else "amostragem com semente fixa"
    print(f"Checkpoint: {args.checkpoint} | device: {model.device} | {modo}")

    resultados = []
    try:
        with criar_ambiente(args.core, args.rom, legado=legado, render=args.render) as env:
            for episode in range(1, args.partidas + 1):
                seed = args.semente + episode - 1
                politica = politica_ppo(model, semente=seed, deterministica=deterministica)
                total, passos, terminou, info = jogar_partida(
                    env, politica, seed=seed,
                )
                resultados.append(total)
                motivo = "game over" if terminou else "limite de passos"
                print(
                    f"Partida {episode}: {motivo} | placar={info['score']} | "
                    f"vidas={info['lives']} | pontos={total:.0f} | passos={passos}"
                )
    except KeyboardInterrupt:
        pass

    if resultados:
        print(
            f"{len(resultados)} partidas: média={mean(resultados):.1f} | "
            f"mediana={median(resultados):.1f} | máximo={max(resultados):.0f}"
        )


if __name__ == "__main__":
    main()
