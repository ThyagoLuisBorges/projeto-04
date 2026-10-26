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
M_MAX_PERFIL = 49
AZUL, LARANJA = "#2a78d6", "#eb6834"


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
    contagem = np.bincount(np.abs(i - j).ravel(), weights=media.ravel(), minlength=m)
    return contagem


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
    cabeca = int(np.argmin(ents))
    media = pesos[:, cabeca].numpy().mean(axis=0)
    m = media.shape[0]
    perfil = perfil_defasagem(media) / m
    if m < M_MAX_PERFIL:
        perfil = np.concatenate([perfil, np.zeros(M_MAX_PERFIL - m)])
    ate = lambda d: float(perfil[: d + 1].sum())
    resumo = {
        "modelo": f"codificador_h{horizonte}",
        "cabeca": cabeca,
        "m": m,
        "entropia": float(ents[cabeca]),
        "entropia_uniforme": float(np.log(m)),
        "massa_ate_1h": ate(1),
        "massa_ate_6h": ate(6),
        "massa_ate_24h": ate(24),
        "massa_exatamente_24h": float(perfil[24]),
    }
    np.savetxt(DADOS / f"atencao_media_h{horizonte}.csv", media, fmt="%.6f", delimiter=",")
    return {"resumo": resumo, "media": media, "perfil": perfil}


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


def figura_perfil(perfis: dict[int, np.ndarray]) -> None:
    _estilo()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    eixo = np.arange(M_MAX_PERFIL)
    for horizonte, cor in ((1, AZUL), (24, LARANJA)):
        ax.plot(eixo, perfis[horizonte][:M_MAX_PERFIL], "-o", color=cor, ms=3.5, lw=1.8,
                label=f"horizonte {horizonte} h")
    ax.set_xlabel("defasagem |consulta − chave| (horas)")
    ax.set_ylabel("massa de atenção")
    ax.set_title("Peso médio por defasagem temporal")
    ax.set_xticks([0, 6, 12, 24, 36, 48])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "atencao_perfil.png", bbox_inches="tight")
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

    resumos, perfis = [], {}
    for horizonte in (1, 24):
        resultado = extrai(horizonte, dados, dev)
        resumo = resultado["resumo"]
        resumos.append(resumo)
        perfis[horizonte] = resultado["perfil"]
        figura_mapa(resultado["media"], horizonte, resumo["cabeca"])
        print(f"horizonte {horizonte:2d} h: cabeça {resumo['cabeca']} "
              f"(entropia {resumo['entropia']:.3f} contra {resumo['entropia_uniforme']:.3f} "
              f"da uniforme), massa até 1 h {resumo['massa_ate_1h']:.3f}, "
              f"até 6 h {resumo['massa_ate_6h']:.3f}, até 24 h {resumo['massa_ate_24h']:.3f}, "
              f"na defasagem 24 h {resumo['massa_exatamente_24h']:.4f}")

    figura_perfil(perfis)
    with open(DADOS / "atencao_resumo.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(resumos[0].keys()))
        w.writeheader()
        w.writerows(resumos)
    with open(DADOS / "atencao_perfil.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["defasagem_h", "massa_h1", "massa_h24"])
        for d in range(min(M_MAX_PERFIL, min(len(p) for p in perfis.values()))):
            w.writerow([d, f"{perfis[1][d]:.6f}", f"{perfis[24][d]:.6f}"])
    print(f"\nSaídas: {DADOS}/atencao_media_h*.csv, {DADOS}/atencao_resumo.csv, "
          f"{DADOS}/atencao_perfil.csv, figs/atencao_*.png")


if __name__ == "__main__":
    main()
