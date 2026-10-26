from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer import (
    DADOS,
    FIGS,
    Codificador,
    _estilo,
    carrega_dados,
    dispositivo,
    janelas,
)

N_AMOSTRAS = 64


def pesos_atencao(modelo: Codificador, x: torch.Tensor, dev) -> torch.Tensor:
    modelo.eval()
    with torch.no_grad():
        _, pesos = modelo(x.to(dev), retorna_pesos=True)
    return pesos.cpu()


def entropias(pesos: torch.Tensor) -> np.ndarray:
    p = pesos.numpy().clip(1e-12, None)
    return (-(p * np.log(p)).sum(axis=-1)).mean(axis=(0, 2))


def perfil_defasagem(media: np.ndarray) -> np.ndarray:
    m = media.shape[0]
    i, j = np.indices((m, m))
    return np.bincount(np.abs(i - j).ravel(), weights=media.ravel(), minlength=m)


def extrai(horizonte: int, dados, dev) -> dict:
    ckpt = torch.load(DADOS / f"codificador_h{horizonte}.pt", weights_only=True)
    cfg = ckpt["config"]
    modelo = Codificador(cfg["m"], cfg["n"], cfg["cabecas"], cfg["blocos"], cfg["horizonte"])
    modelo.load_state_dict(ckpt["estado"])
    modelo = modelo.to(dev)
    x, _ = janelas(dados["teste"], cfg["m"], horizonte=1)
    idx = np.linspace(0, len(x) - 1, N_AMOSTRAS).astype(int)
    pesos = pesos_atencao(modelo, x[idx], dev)
    ents = entropias(pesos)
    m = cfg["m"]
    cabecas = []
    for cabeca in range(cfg["cabecas"]):
        media = pesos[:, cabeca].numpy().mean(axis=0)
        perfil = perfil_defasagem(media) / m
        cabecas.append(
            {
                "modelo": f"codificador_h{horizonte}",
                "cabeca": cabeca,
                "entropia": float(ents[cabeca]),
                "massa_ultimo_token": float(media[:, -1].sum() / m),
                "massa_ate_6h": float(perfil[: min(7, m)].sum()),
            }
        )
    cabeca = int(np.argmin(ents))
    media = pesos[:, cabeca].numpy().mean(axis=0)
    perfil = perfil_defasagem(media) / m
    resumo = {
        "modelo": f"codificador_h{horizonte}",
        "cabeca": cabeca,
        "m": m,
        "entropia": float(ents[cabeca]),
        "entropia_uniforme": float(np.log(m)),
        "massa_ultimo_token": float(media[:, -1].sum() / m),
        "massa_ate_1h": float(perfil[: min(2, m)].sum()),
        "massa_ate_6h": float(perfil[: min(7, m)].sum()),
    }
    np.savetxt(DADOS / f"atencao_media_h{horizonte}.csv", media, fmt="%.6f", delimiter=",")
    return {"resumo": resumo, "media": media, "cabecas": cabecas}


def figura_mapa(media: np.ndarray, horizonte: int, cabeca: int) -> None:
    _estilo()
    m = media.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(media, cmap="viridis", origin="lower")
    passo = max(1, m // 12)
    ax.set_xticks(range(0, m, passo))
    ax.set_yticks(range(0, m, passo))
    ax.set_xlabel("token (chave)")
    ax.set_ylabel("token (consulta)")
    ax.set_title(f"Horizonte {horizonte} h: média dos pesos da cabeça {cabeca}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="peso médio")
    fig.tight_layout()
    fig.savefig(FIGS / f"atencao_mapa_h{horizonte}.png", bbox_inches="tight")
    plt.close(fig)


def testes() -> None:
    torch.manual_seed(0)
    modelo = Codificador(m=6, n=16, cabecas=2, blocos=1, horizonte=1)
    x = torch.randn(3, 6, 14)
    pesos = pesos_atencao(modelo, x, torch.device("cpu"))
    assert pesos.shape == (3, 2, 6, 6)
    assert np.allclose(pesos.sum(dim=-1).numpy(), 1.0, atol=1e-5)

    ents = entropias(pesos)
    assert ents.shape == (2,) and (ents > 0).all() and (ents <= np.log(6) + 1e-6).all()

    perfil = perfil_defasagem(np.full((4, 4), 1 / 16))
    assert np.isclose(perfil.sum(), 1.0)
    assert np.isclose(perfil[0], 4 / 16)
    assert np.isclose(perfil[1], 6 / 16)
    assert np.isclose(perfil[3], 2 / 16)

    perfil_curto = perfil_defasagem(np.full((3, 3), 1 / 9))
    assert np.isclose(perfil_curto.sum(), 1.0) and len(perfil_curto) == 3

    matriz_atencao = torch.rand(4, 4).softmax(dim=-1).numpy()
    fracoes = perfil_defasagem(matriz_atencao) / 4
    assert np.isclose(fracoes.sum(), 1.0)

    medio = torch.rand(5, 2, 4, 4).softmax(dim=-1)[:, 0].mean(axis=0).numpy()
    assert medio.shape == (4, 4)
    print("testes atencao: ok")


def main() -> None:
    testes()
    dev = dispositivo()
    print(f"\ndispositivo: {dev}")
    _, _, dados, _, _, _, _ = carrega_dados()

    resumos, linhas_cabecas = [], []
    for horizonte in (1, 24):
        resultado = extrai(horizonte, dados, dev)
        resumo = resultado["resumo"]
        resumos.append(resumo)
        linhas_cabecas += resultado["cabecas"]
        figura_mapa(resultado["media"], horizonte, resumo["cabeca"])
        print(f"horizonte {horizonte:2d} h: cabeça {resumo['cabeca']} "
              f"(entropia {resumo['entropia']:.3f} contra {resumo['entropia_uniforme']:.3f} "
              f"da uniforme), massa no último token {resumo['massa_ultimo_token']:.3f}")
        for ln in resultado["cabecas"]:
            print(f"  cabeça {ln['cabeca']}: entropia {ln['entropia']:.3f}, "
                  f"último token {ln['massa_ultimo_token']:.3f}, "
                  f"até 6 h {ln['massa_ate_6h']:.3f}")

    with open(DADOS / "atencao_resumo.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(resumos[0].keys()))
        w.writeheader()
        w.writerows(resumos)
    with open(DADOS / "atencao_cabecas.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(linhas_cabecas[0].keys()))
        w.writeheader()
        w.writerows(linhas_cabecas)
    print(f"\nSaídas: {DADOS}/atencao_media_h*.csv, {DADOS}/atencao_resumo.csv, "
          f"{DADOS}/atencao_cabecas.csv, figs/atencao_mapa_h*.png")


if __name__ == "__main__":
    main()
