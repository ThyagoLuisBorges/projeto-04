from __future__ import annotations

import csv
import math
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RAIZ = Path(__file__).resolve().parent
FIGS = RAIZ / "figs"
DADOS = RAIZ / "dados"

SEMENTE = 0
LOTE = 128
TAXA = 1e-3
EPOCAS_BUSCA = 4
EPOCAS = 15
PASSO_BUSCA = 2
MS_BUSCA = [12, 24, 48, 96]
NS_BUSCA = [32, 64]
CABECAS_BUSCA = [2, 4]
BLOCOS_BUSCA = [1, 2]
N_VARIAVEIS = 14
VAR_TEMPERATURA = "T (degC)"

AZUL, LARANJA, AQUA, VERM = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"


def _estilo():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10.5,
            "axes.edgecolor": EIXO,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def dispositivo() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def baixa_dados() -> Path:
    DADOS.mkdir(exist_ok=True)
    caminho = DADOS / "jena_climate_2009_2016.csv"
    if not caminho.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        alvo = DADOS / "jena.zip"
        urllib.request.urlretrieve(url, alvo)
        with zipfile.ZipFile(alvo) as z:
            z.extractall(DADOS)
        alvo.unlink()
    return caminho


def carrega_dados():
    caminho = baixa_dados()
    bruto = pd.read_csv(caminho)
    bruto["Date Time"] = pd.to_datetime(bruto["Date Time"], format="%d.%m.%Y %H:%M:%S")
    bruto = bruto.set_index("Date Time").sort_index()
    vento = ["wv (m/s)", "max. wv (m/s)"]
    bruto[vento] = bruto[vento].clip(lower=0)
    variaveis = list(bruto.columns)
    horaria = bruto.resample("1h").mean().dropna()
    serie = horaria.to_numpy(dtype=np.float32)
    c1, c2 = int(len(serie) * 0.7), int(len(serie) * 0.8)
    partes = {
        "treino": serie[:c1],
        "val": serie[c1:c2],
        "teste": serie[c2:],
    }
    media = serie[:c1].mean(axis=0)
    desvio = serie[:c1].std(axis=0)
    padronizadas = {k: (v - media) / desvio for k, v in partes.items()}
    return bruto, horaria, padronizadas, media, desvio, variaveis, (c1, c2)


def janelas(matriz: np.ndarray, m: int, horizonte: int, passo: int = 1):
    w = np.lib.stride_tricks.sliding_window_view(matriz, m + horizonte, axis=0)[::passo]
    x = w[:, :, :m].transpose(0, 2, 1).copy()
    y = w[:, :, m:].transpose(0, 2, 1).copy()
    return torch.from_numpy(x), torch.from_numpy(y)


def carregadores(m: int, horizonte: int, dados, passo_treino: int = 1, lote: int = LOTE):
    tensores = {}
    for nome, matriz in dados.items():
        x, y = janelas(matriz, m, horizonte, passo=passo_treino if nome == "treino" else 1)
        tensores[nome] = TensorDataset(x, y)
    g = torch.Generator().manual_seed(SEMENTE)
    return {
        "treino": DataLoader(tensores["treino"], batch_size=lote, shuffle=True, generator=g),
        "val": DataLoader(tensores["val"], batch_size=lote * 4),
        "teste": DataLoader(tensores["teste"], batch_size=lote * 4),
    }


class CodificacaoPosicional(nn.Module):
    def __init__(self, n: int, m_max: int = 512):
        super().__init__()
        pe = torch.zeros(m_max, n)
        pos = torch.arange(m_max, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, n, 2, dtype=torch.float32) * (-math.log(10000.0) / n))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class AtencaoMultipla(nn.Module):
    def __init__(self, n: int, cabecas: int):
        super().__init__()
        self.cabecas = cabecas
        self.dh = n // cabecas
        self.consulta = nn.Linear(n, n)
        self.chave = nn.Linear(n, n)
        self.valor = nn.Linear(n, n)
        self.saida = nn.Linear(n, n)

    def forward(self, x: torch.Tensor, retorna_pesos: bool = False):
        b, m, n = x.shape
        partes = []
        for camada in (self.consulta, self.chave, self.valor):
            t = camada(x).reshape(b, m, self.cabecas, self.dh).transpose(1, 2)
            partes.append(t)
        q, k, v = partes
        pesos = (q @ k.transpose(-2, -1) / self.dh**0.5).softmax(dim=-1)
        ctx = (pesos @ v).transpose(1, 2).reshape(b, m, n)
        saida = self.saida(ctx)
        if retorna_pesos:
            return saida, pesos
        return saida


class BlocoCodificador(nn.Module):
    def __init__(self, n: int, cabecas: int):
        super().__init__()
        self.atencao = AtencaoMultipla(n, cabecas)
        self.norma1 = nn.LayerNorm(n)
        self.norma2 = nn.LayerNorm(n)
        self.ffn = nn.Sequential(nn.Linear(n, 4 * n), nn.ReLU(), nn.Linear(4 * n, n))

    def forward(self, x: torch.Tensor, retorna_pesos: bool = False):
        if retorna_pesos:
            atencao, pesos = self.atencao(x, retorna_pesos=True)
        else:
            atencao = self.atencao(x)
        x = self.norma1(x + atencao)
        x = self.norma2(x + self.ffn(x))
        if retorna_pesos:
            return x, pesos
        return x


class Codificador(nn.Module):
    def __init__(self, m: int, n: int, cabecas: int, blocos: int, horizonte: int):
        super().__init__()
        self.horizonte = horizonte
        self.projecao = nn.Linear(N_VARIAVEIS, n)
        self.posicao = CodificacaoPosicional(n)
        self.blocos = nn.ModuleList([BlocoCodificador(n, cabecas) for _ in range(blocos)])
        self.cabeca = nn.Linear(n, horizonte * N_VARIAVEIS)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor, retorna_pesos: bool = False):
        h = self.posicao(self.projecao(x))
        pesos = None
        for bloco in self.blocos:
            if retorna_pesos:
                h, pesos = bloco(h, retorna_pesos=True)
            else:
                h = bloco(h)
        rep = h.mean(dim=1)
        valores = self.cabeca(rep).reshape(x.size(0), self.horizonte, N_VARIAVEIS)
        if retorna_pesos:
            return valores, pesos
        return valores


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def treina_epoca(modelo, carregador, otimizador, perda, dev) -> float:
    modelo.train()
    total, n = 0.0, 0
    for x, y in carregador:
        x, y = x.to(dev), y.to(dev)
        otimizador.zero_grad()
        l = perda(modelo(x), y)
        l.backward()
        otimizador.step()
        total += l.item() * len(y)
        n += len(y)
    return total / n


def avalia(modelo, carregador, perda, dev) -> float:
    modelo.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in carregador:
            x, y = x.to(dev), y.to(dev)
            total += perda(modelo(x), y).item() * len(y)
            n += len(y)
    return total / n


def treina_modelo(modelo, carregadores, epocas, dev, taxa=TAXA):
    modelo = modelo.to(dev)
    perda = nn.MSELoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa)
    curva_treino, curva_val = [], []
    t0 = time.perf_counter()
    for _ in range(epocas):
        curva_treino.append(treina_epoca(modelo, carregadores["treino"], otimizador, perda, dev))
        curva_val.append(avalia(modelo, carregadores["val"], perda, dev))
    tempo = time.perf_counter() - t0
    return modelo, curva_treino, curva_val, tempo


def resumo_dados(bruto: pd.DataFrame, horaria: pd.DataFrame, variaveis: list[str]) -> None:
    _estilo()
    FIGS.mkdir(exist_ok=True)
    DADOS.mkdir(exist_ok=True)
    linhas = []
    for v in variaveis:
        s = bruto[v]
        linhas.append(
            {"variavel": v, "media": s.mean(), "desvio": s.std(), "minimo": s.min(), "maximo": s.max()}
        )
    with open(DADOS / "estatisticas.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["variavel", "media", "desvio", "minimo", "maximo"])
        w.writeheader()
        w.writerows(linhas)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2))
    ax1.plot(horaria.index, horaria[VAR_TEMPERATURA], color=AZUL, lw=0.4)
    ax1.set_ylabel("T (°C)")
    ax1.set_title("Temperatura do ar por hora, Jena, 2009–2016")
    recorte = horaria[VAR_TEMPERATURA]["2015-06-01":"2015-06-14"]
    ax2.plot(recorte.index, recorte, color=LARANJA, lw=1.0)
    ax2.set_ylabel("T (°C)")
    ax2.set_title("Detalhe de duas semanas (1–14 de junho de 2015)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "serie.png", bbox_inches="tight")
    plt.close(fig)
    print(f"linhas a cada 10 min: {len(bruto)}, horas: {len(horaria)}")
    print(f"{'variavel':>15} {'media':>10} {'desvio':>10} {'minimo':>10} {'maximo':>10}")
    for ln in linhas:
        print(f"{ln['variavel']:>15} {ln['media']:>10.3f} {ln['desvio']:>10.3f} "
              f"{ln['minimo']:>10.3f} {ln['maximo']:>10.3f}")


def busca_hiperparametros(dados, dev) -> dict:
    resultados = []

    def roda(m, n, cabecas, blocos, etapa):
        torch.manual_seed(SEMENTE)
        modelo = Codificador(m, n, cabecas, blocos, horizonte=1)
        carreg = carregadores(m, 1, dados, passo_treino=PASSO_BUSCA)
        _, _, curva_val, tempo = treina_modelo(modelo, carreg, EPOCAS_BUSCA, dev)
        resultados.append(
            {
                "etapa": etapa,
                "m": m,
                "n": n,
                "cabecas": cabecas,
                "blocos": blocos,
                "parametros": n_parametros(modelo),
                "perda_val": curva_val[-1],
                "tempo_s": tempo,
                "epocas": EPOCAS_BUSCA,
            }
        )
        print(f"  m={m:3d} n={n:3d} cabeças={cabecas} blocos={blocos}: "
              f"perda val {curva_val[-1]:.4f}, {tempo:.1f} s")

    print(f"etapa 1: contexto m (n=32, cabeças=4, blocos=1), {EPOCAS_BUSCA} épocas")
    for m in MS_BUSCA:
        roda(m, 32, 4, 1, "contexto")
    m_escolhido = min(
        (r for r in resultados if r["etapa"] == "contexto"), key=lambda r: r["perda_val"]
    )["m"]
    print(f"etapa 2: arquitetura (m={m_escolhido})")
    for n in NS_BUSCA:
        for cabecas in CABECAS_BUSCA:
            for blocos in BLOCOS_BUSCA:
                roda(m_escolhido, n, cabecas, blocos, "arquitetura")

    with open(DADOS / "busca.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    arquitetura = [r for r in resultados if r["etapa"] == "arquitetura"]
    melhor = min(arquitetura, key=lambda r: (r["perda_val"], r["parametros"]))
    return melhor


def previsoes_lote(modelo, carregador, dev):
    modelo.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in carregador:
            ys.append(y)
            ps.append(modelo(x.to(dev)).cpu())
    return torch.cat(ys), torch.cat(ps)


def mae_por_variavel(y_real: torch.Tensor, y_pred: torch.Tensor, desvio: np.ndarray) -> np.ndarray:
    return (y_real - y_pred).abs().mean(dim=(0, 1)).numpy() * desvio


def persistencia(carregador):
    xs, ys = [], []
    for x, y in carregador:
        xs.append(x[:, -1, :])
        ys.append(y)
    x = torch.cat(xs).unsqueeze(1)
    y = torch.cat(ys)
    pred = x.expand(-1, y.size(1), -1)
    return y, pred


def figura_perda(curva_treino, curva_val, caminho, titulo) -> None:
    _estilo()
    epocas = np.arange(1, len(curva_treino) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(epocas, curva_treino, "-o", color=AZUL, ms=4, lw=2, label="treino")
    ax.plot(epocas, curva_val, "-s", color=LARANJA, ms=4, lw=2, label="validação")
    ax.set_xlabel("época")
    ax.set_ylabel("perda (EQM padronizado)")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


def figura_previsao_h1(horaria, c2, m, y_real, y_pred, media, desvio, variaveis) -> None:
    _estilo()
    it = variaveis.index(VAR_TEMPERATURA)
    inicio = 24
    fim = inicio + 24 * 7
    real = y_real[inicio:fim, 0, it].numpy() * desvio[it] + media[it]
    pred = y_pred[inicio:fim, 0, it].numpy() * desvio[it] + media[it]
    idx = horaria.index[c2 + m + inicio : c2 + m + fim]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(idx, real, color=AZUL, lw=1.4, label="observado")
    ax.plot(idx, pred, color=LARANJA, lw=1.2, ls="--", label="previsto")
    ax.set_ylabel("T (°C)")
    ax.set_title("Uma hora à frente: primeira semana do teste")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "previsao_h1.png", bbox_inches="tight")
    plt.close(fig)


def figura_previsao_h24(horaria, c2, m, y_real, y_pred, media, desvio, variaveis) -> None:
    _estilo()
    it = variaveis.index(VAR_TEMPERATURA)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    esquerdas = [24 * i for i in range(6)]
    for j, esq in enumerate(esquerdas):
        real = y_real[esq, :, it].numpy() * desvio[it] + media[it]
        pred = y_pred[esq, :, it].numpy() * desvio[it] + media[it]
        idx = horaria.index[c2 + m + esq : c2 + m + esq + 24]
        ax.plot(idx, real, color=AZUL, lw=1.2, label="observado" if j == 0 else None)
        ax.plot(idx, pred, color=LARANJA, lw=1.2, ls="--", label="previsto" if j == 0 else None)
    ax.set_ylabel("T (°C)")
    ax.set_xlabel("seis janelas consecutivas de 24 h a partir do início do teste")
    ax.set_title("Vinte e quatro horas à frente: seis janelas do teste")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "previsao_h24.png", bbox_inches="tight")
    plt.close(fig)


def experimento_final(config: dict, dados, horaria, c2, media, desvio, variaveis, dev) -> dict:
    m, n, cabecas, blocos = config["m"], config["n"], config["cabecas"], config["blocos"]
    resumo = []
    for horizonte in (1, 24):
        torch.manual_seed(SEMENTE)
        modelo = Codificador(m, n, cabecas, blocos, horizonte)
        carreg = carregadores(m, horizonte, dados)
        modelo, curva_treino, curva_val, tempo = treina_modelo(modelo, carreg, EPOCAS, dev)
        figura_perda(curva_treino, curva_val, FIGS / f"perda_h{horizonte}.png",
                     f"Horizonte {horizonte} h: perda por época")
        with open(DADOS / f"curva_h{horizonte}.csv", "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["epoca", "perda_treino", "perda_val"])
            for e, (a, b) in enumerate(zip(curva_treino, curva_val), 1):
                w.writerow([e, f"{a:.6f}", f"{b:.6f}"])
        perda = nn.MSELoss()
        l_teste = avalia(modelo, carreg["teste"], perda, dev)
        y_real, y_pred = previsoes_lote(modelo, carreg["teste"], dev)
        y_perc, pred_perc = persistencia(carreg["teste"])
        it = variaveis.index(VAR_TEMPERATURA)
        mae_t = float(mae_por_variavel(y_real, y_pred, desvio)[it])
        mae_t_perc = float(mae_por_variavel(y_perc, pred_perc, desvio)[it])
        l_perc = perda(pred_perc, y_perc).item()
        resultado = {
            "modelo": f"codificador_h{horizonte}",
            "m": m,
            "n": n,
            "cabecas": cabecas,
            "blocos": blocos,
            "parametros": n_parametros(modelo),
            "epocas": EPOCAS,
            "tempo_total_s": tempo,
            "tempo_epoca_s": tempo / EPOCAS,
            "perda_teste": l_teste,
            "perda_persistencia": l_perc,
            "mae_T": mae_t,
            "mae_T_persistencia": mae_t_perc,
        }
        with open(DADOS / f"resultado_h{horizonte}.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(resultado.keys()))
            w.writeheader()
            w.writerow(resultado)
        torch.save(
            {"config": {"m": m, "n": n, "cabecas": cabecas, "blocos": blocos,
                        "horizonte": horizonte},
             "estado": modelo.state_dict()},
            DADOS / f"codificador_h{horizonte}.pt",
        )
        resumo.append((horizonte, y_real, y_pred, mae_t, mae_t_perc, l_teste, l_perc, tempo))
        print(f"horizonte {horizonte:2d} h: perda teste {l_teste:.4f} "
              f"(persistência {l_perc:.4f}), MAE T {mae_t:.3f} °C "
              f"(persistência {mae_t_perc:.3f} °C), {tempo:.1f} s")

        maes = mae_por_variavel(y_real, y_pred, desvio)
        maes_perc = mae_por_variavel(y_perc, pred_perc, desvio)
        with open(DADOS / f"mae_variaveis_h{horizonte}.csv", "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["variavel", "modelo", "persistencia"])
            for v, a, b in zip(variaveis, maes, maes_perc):
                w.writerow([v, f"{a:.4f}", f"{b:.4f}"])

    figura_previsao_h1(horaria, c2, m, resumo[0][1], resumo[0][2], media, desvio, variaveis)
    figura_previsao_h24(horaria, c2, m, resumo[1][1], resumo[1][2], media, desvio, variaveis)
    return resumo


def testes() -> None:
    torch.manual_seed(SEMENTE)
    modelo = Codificador(m=6, n=16, cabecas=2, blocos=2, horizonte=3)
    x = torch.randn(4, 6, N_VARIAVEIS)
    saida, pesos = modelo(x, retorna_pesos=True)
    assert saida.shape == (4, 3, N_VARIAVEIS)
    assert pesos.shape == (4, 2, 6, 6)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(4, 2, 6), atol=1e-5)

    atencao = (16 * 16 + 16) * 4
    normas = (16 + 16) * 2
    ffn = (16 * 64 + 64) + (64 * 16 + 16)
    esperado = (N_VARIAVEIS * 16 + 16) + 2 * (atencao + normas + ffn) + (16 * 42 + 42)
    assert n_parametros(modelo) == esperado

    matriz = np.arange(5 * N_VARIAVEIS, dtype=np.float32).reshape(5, N_VARIAVEIS)
    x, y = janelas(matriz, m=3, horizonte=2)
    assert x.shape == (1, 3, N_VARIAVEIS) and y.shape == (1, 2, N_VARIAVEIS)
    assert np.allclose(x[0].numpy(), matriz[:3]) and np.allclose(y[0].numpy(), matriz[3:5])

    pe = CodificacaoPosicional(8)
    assert pe(torch.zeros(1, 4, 8)).shape == (1, 4, 8)
    assert torch.allclose(pe.pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe.pe[0, 1::2], torch.ones(4))

    lote_x = torch.randn(8, 6, N_VARIAVEIS)
    lote_y = torch.randn(8, 3, N_VARIAVEIS)
    pequeno = Codificador(6, 16, 2, 1, 3)
    otimizador = torch.optim.AdamW(pequeno.parameters(), lr=1e-2)
    perda = nn.MSELoss()
    perdas = []
    for _ in range(40):
        otimizador.zero_grad()
        l = perda(pequeno(lote_x), lote_y)
        l.backward()
        otimizador.step()
        perdas.append(l.item())
    assert perdas[-1] < perdas[0]
    print("testes transformer: ok")


def main() -> None:
    testes()
    dev = dispositivo()
    print(f"\ndispositivo: {dev}")
    bruto, horaria, dados, media, desvio, variaveis, (c1, c2) = carrega_dados()
    print(f"horas: treino {c1}, val {c2 - c1}, teste {len(horaria) - c2}")

    print("\nDados:")
    resumo_dados(bruto, horaria, variaveis)

    print(f"\nBusca de hiperparâmetros ({EPOCAS_BUSCA} épocas, passo {PASSO_BUSCA}):")
    melhor = busca_hiperparametros(dados, dev)
    print(f"escolhido: m={melhor['m']}, n={melhor['n']}, "
          f"cabeças={melhor['cabecas']}, blocos={melhor['blocos']}")

    print(f"\nExperimento final ({EPOCAS} épocas, passo 1):")
    experimento_final(melhor, dados, horaria, c2, media, desvio, variaveis, dev)
    print(f"\nSaídas: {DADOS}/busca.csv, {DADOS}/resultado_h*.csv, "
          f"{DADOS}/mae_variaveis_h*.csv, {DADOS}/estatisticas.csv, "
          f"{DADOS}/codificador_h*.pt, figs/*.png")


if __name__ == "__main__":
    main()
