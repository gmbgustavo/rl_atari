import argparse
import csv
import signal
from functools import partial
from pathlib import Path
from statistics import mean, median

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from space_invaders_env import (
    ACAO_TIRO,
    PRIMEIRA_SEMENTE_AVALIACAO,
    criar_ambiente,
    jogar_partida,
)

SAIDA_PADRAO = Path("checkpoints/ppo_sinvaders_v2")
SEMENTES_AVALIACAO = range(PRIMEIRA_SEMENTE_AVALIACAO, PRIMEIRA_SEMENTE_AVALIACAO + 10)
ESCALA_RECOMPENSA = 100.0


def salvar(modelo, caminho):
    temporario = caminho.with_suffix(".tmp.zip")
    try:
        modelo.save(temporario)
        temporario.replace(caminho)
    finally:
        temporario.unlink(missing_ok=True)


def politica_ppo(modelo, *, semente, deterministica=False):
    rng = np.random.default_rng(semente)

    def escolher_acao(obs):
        if deterministica:
            return modelo.predict(obs, deterministic=True)[0].item()
        # Use um gerador próprio: avaliar não altera a sequência de ações do treino.
        with torch.inference_mode():
            tensor, _ = modelo.policy.obs_to_tensor(obs)
            distribuicao = modelo.policy.get_distribution(tensor).distribution
            probabilidades = distribuicao.probs[0].cpu().numpy()
        return rng.choice(len(probabilidades), p=probabilidades)

    modelo.policy.set_training_mode(False)
    return escolher_acao


class AcompanharTreino(BaseCallback):
    def __init__(self, env, saida, intervalo):
        super().__init__()
        self.env = env
        self.intervalo = intervalo
        self.ultimo_passo = None
        self.interrompido = False
        self.latest = saida / "latest.zip"
        self.best = saida / "best.zip"
        self.csv = saida / "avaliacoes.csv"

    def _on_training_start(self):
        if self.csv.is_file() and self.best.is_file():
            self.ultimo_passo = getattr(self.model, "ultima_avaliacao", None)
        self.avaliar()

    def _on_rollout_start(self):
        # Neste ponto, o PPO já atualizou a política com o rollout anterior.
        if self.num_timesteps - self.ultimo_passo >= self.intervalo:
            self.avaliar()

    def _on_step(self):
        return not self.interrompido

    def solicitar_interrupcao(self, signum, frame):
        # Não interrompa uma leitura parcial das conexões do SubprocVecEnv.
        self.interrompido = True

    def medir_partida(self, politica, seed):
        if self.interrompido:
            raise KeyboardInterrupt
        return jogar_partida(self.env, politica, seed=seed)[0]

    def avaliar(self):
        passos = self.model.num_timesteps
        if passos == self.ultimo_passo:
            return

        resultados = [
            self.medir_partida(politica_ppo(self.model, semente=seed), seed)
            for seed in SEMENTES_AVALIACAO
        ]
        media = mean(resultados)
        referencia = getattr(self.model, "referencia_avaliacao", None)
        if referencia is None:
            resultados_aleatorios = []
            for seed in SEMENTES_AVALIACAO:
                rng = np.random.default_rng(seed)
                pontos = self.medir_partida(
                    lambda obs: rng.integers(self.env.action_space.n), seed,
                )
                resultados_aleatorios.append(pontos)
            tiro_fixo = [
                self.medir_partida(lambda obs: ACAO_TIRO, seed)
                for seed in SEMENTES_AVALIACAO
            ]
            referencia = {
                "inicial": media,
                "aleatorio": mean(resultados_aleatorios),
                "tiro_fixo": mean(tiro_fixo),
            }
            self.model.referencia_avaliacao = referencia

        self.model.ultima_avaliacao = passos
        novo_melhor = not self.best.is_file() or media > referencia.get("melhor", -np.inf)
        if novo_melhor:
            referencia["melhor"] = media
            salvar(self.model, self.best)
        salvar(self.model, self.latest)

        linha = {
            "passos": passos,
            "partidas": len(resultados),
            "media": media,
            "mediana": median(resultados),
            "maximo": max(resultados),
            "media_inicial": referencia["inicial"],
            "media_aleatoria": referencia["aleatorio"],
            "media_tiro_fixo": referencia["tiro_fixo"],
            "melhor_media": referencia["melhor"],
        }
        cabecalho = not self.csv.exists() or self.csv.stat().st_size == 0
        with self.csv.open("a", encoding="utf-8", newline="") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=list(linha))
            if cabecalho:
                writer.writeheader()
            writer.writerow(linha)

        destaque = " | novo melhor" if novo_melhor else ""
        print(
            f"[avaliação | {passos:,} passos] média={media:.1f} "
            f"({media - referencia['inicial']:+.1f} vs. inicial) | "
            f"mediana={linha['mediana']:.1f} | máximo={linha['maximo']:.0f} | "
            f"aleatório={referencia['aleatorio']:.1f} "
            f"({media - referencia['aleatorio']:+.1f} vs. aleatório) | "
            f"tiro fixo={referencia['tiro_fixo']:.1f} | "
            f"melhor média={referencia['melhor']:.1f}{destaque}",
            flush=True,
        )
        self.ultimo_passo = passos


def ambiente_treino(core, rom):
    # O processo principal salva o progresso e fecha os workers no Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = Monitor(criar_ambiente(core, rom))
    # Preserve a proporção entre pontuações e registre os pontos brutos no Monitor.
    return gym.wrappers.TransformReward(env, lambda pontos: pontos / ESCALA_RECOMPENSA)


def treinar(args, env, avaliacao):
    latest = args.saida / "latest.zip"
    retomando = latest.is_file()
    if retomando:
        modelo = PPO.load(latest, env=env, device=args.device)
        print(f"Retomando {latest} ({modelo.num_timesteps:,} passos).", flush=True)
    else:
        modelo = PPO(
            "CnnPolicy", env, device=args.device, seed=args.semente, verbose=1,
            n_steps=128, batch_size=128, n_epochs=4,
            learning_rate=1e-4 if args.iniciar_de else 2.5e-4,
            clip_range=0.1, ent_coef=0.01, target_kl=0.03,
        )
        modelo.space_invaders_v2 = True
        if args.iniciar_de:
            origem = PPO.load(args.iniciar_de, device="cpu")
            # Reaproveite a política, com otimizador novo para a dinâmica/recompensa atual.
            modelo.policy.load_state_dict(origem.policy.state_dict())
            if not getattr(origem, "space_invaders_v2", False):
                # O crítico antigo estimava retornos em pontos brutos.
                with torch.no_grad():
                    modelo.policy.value_net.weight.div_(ESCALA_RECOMPENSA)
                    modelo.policy.value_net.bias.div_(ESCALA_RECOMPENSA)
            print(f"Pesos iniciais carregados de {args.iniciar_de}.", flush=True)

    acompanhamento = AcompanharTreino(avaliacao, args.saida, args.avaliar_a_cada)
    sinal_anterior = signal.signal(signal.SIGINT, acompanhamento.solicitar_interrupcao)
    try:
        modelo.learn(
            total_timesteps=args.passos,
            callback=acompanhamento,
            reset_num_timesteps=not retomando,
            log_interval=10,
        )
        if acompanhamento.interrompido:
            raise KeyboardInterrupt
        acompanhamento.avaliar()
    except KeyboardInterrupt:
        salvar(modelo, latest)
        print(f"\nTreino interrompido. Checkpoint salvo em {latest}.")
    else:
        print(f"Treino concluído. Último: {latest} | Melhor: {acompanhamento.best}")
    finally:
        signal.signal(signal.SIGINT, sinal_anterior)


def main(*, capturar=False):
    parser = argparse.ArgumentParser(description="Treina PPO e acompanha a pontuação em Space Invaders.")
    parser.add_argument("core", type=Path, help="Caminho para o core Stella (.so ou .dll)")
    parser.add_argument("rom", type=Path, help="Caminho para a ROM de Space Invaders")
    parser.add_argument("--passos", type=int, default=500_000, help="Passos adicionais de treino")
    parser.add_argument("--avaliar-a-cada", type=int, default=10_000, help="Passos entre avaliações")
    parser.add_argument("--ambientes", type=int, default=4, help="Emuladores de treino em paralelo")
    parser.add_argument("--threads", type=int, default=2, help="Threads de CPU para o PyTorch")
    parser.add_argument("--semente", type=int, default=42, help="Semente de um treino novo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO, help="Diretório da execução; retoma latest.zip se existir")
    parser.add_argument("--iniciar-de", type=Path, help="Inicializa um treino novo com os pesos de um checkpoint")
    args = parser.parse_args()

    if min(args.passos, args.avaliar_a_cada, args.ambientes, args.threads) < 1:
        parser.error("--passos, --avaliar-a-cada, --ambientes e --threads devem ser positivos")
    if not 0 <= args.semente <= 2**32 - args.ambientes:
        parser.error("--semente fora do intervalo aceito pelo NumPy")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponível; use --device auto ou cpu")
    for caminho in (args.core, args.rom, args.iniciar_de):
        if caminho is not None and not caminho.is_file():
            parser.error(f"Arquivo não encontrado: {caminho}")
    if args.iniciar_de and (args.saida / "latest.zip").exists():
        parser.error("--iniciar-de requer um novo diretório em --saida")
    torch.set_num_threads(args.threads)
    args.saida.mkdir(parents=True, exist_ok=True)

    with criar_ambiente(args.core, args.rom) as avaliacao:
        avaliacao.reset(seed=PRIMEIRA_SEMENTE_AVALIACAO)
        if capturar:
            salvar_frame(avaliacao, "INICIO.png")
        fabrica = partial(ambiente_treino, args.core, args.rom)
        # O core Libretro mantém estado global; cada emulador roda em um processo.
        env = SubprocVecEnv([fabrica] * args.ambientes, start_method="spawn")
        try:
            treinar(args, env, avaliacao)
        finally:
            env.close()
        if capturar:
            salvar_frame(avaliacao, "FIM_TESTE.png")
            salvar_frame(avaliacao, "frame.png")


def salvar_frame(env, nome):
    from PIL import Image

    Image.fromarray(env.unwrapped.stella.rgb()).save(nome)
    print(f"Frame salvo em {nome}.")


if __name__ == "__main__":
    main()
