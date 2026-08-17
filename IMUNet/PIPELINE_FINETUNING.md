# Pipeline: dados brutos → fine-tuning → teste

Este documento registra o passo a passo do que foi feito nesta sessão: partir do checkpoint
pré-treinado do RONIN, fazer fine-tuning com o dataset publicado do IMUNet
(`Datasets/proposed/IMUNet_dataset`), e avaliar o resultado num conjunto de teste
nunca visto. Serve de referência pra repetir o processo com o seu próprio dataset de 20h.

Fluxo completo (incluindo o passo que ainda falta rodar):

```
sensores brutos (.txt)  →  read_data_s10.py  →  processed/data.csv  →  fine-tuning (train)  →  teste/plot
     [seu dataset]           [não rodamos          [IMUNet_dataset          [main.py via         [main.py via
                              nesta sessão]          já vinha assim]         finetune_driver.py]  test_driver.py]
```

---

## 0. Onde está cada coisa

- Fork de treino: `IMUNet/RONIN_torch/` (`main.py`, `utils.py`, `model_resnet1d.py`)
- Repo original do RONIN (usado só pra comparação): `ronin/source/`
- Dataset publicado do IMUNet usado nesta sessão: `IMUNet/Datasets/proposed/IMUNet_dataset/`
  (126 sequências, `list_train.txt` com 90, `list_test.txt` com 36)
- Checkpoint pré-treinado do RONIN (baixado pelo usuário): `IMUNet/Datasets/ronin/ronin_resnet_model/checkpoint_gsn_latest.pt`
  (ResNet1D, época 100, arquitetura confirmada 100% compatível com `model_resnet1d.py` deste repo)

---

## 1. Dados brutos (.txt) → `processed/data.csv` — `read_data_s10.py`

**Importante**: das 126 sequências em `IMUNet_dataset/`, só **1** (`Outdoor_Subject_1_Tango_6`) tem os
`.txt` brutos — as outras 125 já vêm só com `processed/`. Ou seja, o autor original do IMUNet
publicou majoritariamente o dado já processado; esse passo é o que **você** vai precisar rodar
pro seu próprio dataset de 20h, que ainda está só em `.txt`.

### O que o script faz (`IMUNet/Datasets/proposed/read_data_s10.py`)

Lê os sensores brutos de uma pasta de sequência:

| arquivo | conteúdo | formato de cada linha |
|---|---|---|
| `pose.txt` | pose do ARCore (groundtruth) | `timestamp_ns  pos_x pos_y pos_z  quat_x quat_y quat_z quat_w` |
| `acce.txt` | acelerômetro | `timestamp_ns  acce_x acce_y acce_z` |
| `gyro.txt` | giroscópio | `timestamp_ns  gyro_x gyro_y gyro_z` |
| `linacce.txt` | aceleração linear (sem gravidade) | idem |
| `gravity.txt` | vetor gravidade | idem |
| `magnet.txt` | magnetômetro | idem |
| `orientation.txt` | rotation vector | `timestamp_ns  quat_x quat_y quat_z quat_w` |

E gera `processed/data.csv` reamostrando tudo pra 200Hz numa timeline única, ancorada nos
timestamps do `pose.txt` (interpolação linear pros vetores, SLERP pros quaternions), com as
colunas: `time,gyro_x,y,z,acce_x,y,z,linacce_x,y,z,grav_x,y,z,magnet_x,y,z,pos_x,y,z,ori_w,x,y,z,rv_w,x,y,z`.

### ⚠️ Ponto crítico pro seu dataset

O script só lê **uma** fonte de rotation vector (`orientation.txt` → vira as colunas `rv_*`).
Como você coleta `rv` **e** `game_rv` separadamente, esse script **não sabe lidar com o
`game_rv`** do jeito que está — se você rodar sem alterar nada, o sinal de `game_rv` do seu
dataset bruto será **descartado silenciosamente** e o `processed/data.csv` só vai ter `rv_*`,
igual ao dataset do autor original. Antes de rodar no seu dataset, o script precisa ganhar um
bloco a mais (análogo ao de `orientation.txt`) que leia o arquivo bruto do `game_rv` (o nome do
arquivo depende de como seu app grava) e escreva `game_rv_w,game_rv_x,game_rv_y,game_rv_z` no
CSV final. Isso ainda não foi feito.

### Como rodar (quando formos fazer isso pro seu dataset)

```bash
cd IMUNet/Datasets/proposed   # ou onde estiver seu dataset

# uma sequência por vez (respeita --path normalmente):
python read_data_s10.py --path /caminho/completo/da/sequencia

# lote (CUIDADO: o script ignora o valor passado em --list e sempre procura
# um arquivo chamado exatamente "list_data_s10.txt" no diretório onde você
# roda o comando — é o mesmo tipo de hardcode que já vimos em main.py):
#   list_data_s10.txt: um nome de pasta de sequência por linha
python read_data_s10.py
```

Flags úteis: `--skip_front 120 --skip_end 120` (default, descarta as primeiras/últimas 120
amostras de pose — calibração/estabilização do ARCore no início/fim), `--recompute` (força
reprocessar mesmo se `processed/data.csv` já existir), `--no_magnet` (pula magnetômetro se
os dados estiverem corrompidos).

---

## 2. Parametrização do preprocessamento — `ProposedSequence`

Antes de treinar, adicionamos dois parâmetros em `ProposedSequence` (`IMUNet/RONIN_torch/utils.py`)
e nas flags de CLI de `main.py`, porque o código original tinha isso tudo fixo:

- **`--ori_source {rv, game_rv}`** (default `rv`): qual coluna usar como fonte de orientação.
  O RONIN original nunca usa `rv` puro (com magnetômetro) — só `game_rv`. `rv` é suscetível a
  ruído/salto por interferência magnética em tempo real (elevador, estrutura metálica, etc.),
  o que é candidato a explicar a "alucinação" observada no modelo IMUNet anterior.
- **`--no_init_rotor`** (default: init_rotor **ligado**): o `init_rotor` alinha a orientação ao
  frame absoluto do ARCore usando o `ori_w/x/y/z` do primeiro frame — um dado que só existe
  offline. Ele **precisa** ficar ligado no treino (é o que garante que a feature e o target
  estejam no mesmo referencial — sem isso, cada sequência de treino tem um desalinhamento de
  heading diferente e não aprendível), e é justamente **removido na inferência em tempo real**
  (que é o que o app já faz). Esse "descasamento" treino/inferência é intencional — é
  exatamente o que o `RandomHoriRotate(2π)` (já existente no pipeline, roda só no treino) treina
  a rede pra tolerar. Confirmamos que o RONIN original segue o mesmo padrão.

O dataset `IMUNet_dataset` só tem coluna `rv_*` (sem `game_rv_*`), então o fine-tuning desta
sessão só pôde testar a segunda parte (`init_rotor` ligado no treino) — a comparação `rv` vs
`game_rv` depende do seu dataset.

---

## 3. Ambiente — bugs corrigidos pra conseguir treinar

`main.py` tinha vários problemas que só não apareciam rodando do jeito "normal" (`python main.py`
como script principal) ou com uma versão de PyTorch mais antiga. Todos em `IMUNet/RONIN_torch/`:

| arquivo | problema | correção |
|---|---|---|
| `main.py` | `from CNN_LSTM import *` (módulo inexistente neste checkout) incondicional no topo | import movido pra dentro do branch `arch == 'LSTM'` (lazy) |
| `main.py` | `get_model(arch)` ignorava o parâmetro e lia uma variável global `args` | parâmetro renomeado pra `args`, usado corretamente |
| `main.py` | `run_test()` também lia `args` global (mesmo padrão) | driver scripts fazem `main.args = args` antes de chamar `train`/`test_sequence` |
| `main.py` | `ReduceLROnPlateau(..., verbose=True)` — kwarg removido no PyTorch 2.x | `verbose` removido (só afetava log, não a matemática) |
| `main.py` | `from torchsummary import summary` importado e nunca usado, pacote não instalado | linha removida (código morto) |
| `main.py` | `--continue_from` restaurava o `optimizer_state_dict` inteiro, **sobrescrevendo** silenciosamente qualquer `--lr` novo passado pra fine-tuning | depois do `load_state_dict`, reaplica `args.lr` em `optimizer.param_groups` |
| ambiente | `onnx` não instalado (`torch.onnx.export` é chamado de verdade em `test_sequence`, não é código morto) | `pip install onnx` |
| `main.py` (`if __name__=='__main__':`) | sobrescreve `root_dir`/`train_list`/`val_list`/`out_dir`/`model_path`/`test_list` incondicionalmente com base em `--dataset`, ignorando qualquer valor custom passado via CLI | contornado com scripts próprios (`finetune_driver.py`, `test_driver.py`) que chamam `train()`/`test_sequence()` direto, sem passar pelo bloco `__main__` |

---

## 4. Fine-tuning (treino)

Script: `IMUNet/RONIN_torch/finetune_driver.py` (constrói os `args` manualmente e chama
`train()` direto, contornando o hardcode de paths do `main.py`).

Configuração usada:

```python
continue_from = 'IMUNet/Datasets/ronin/ronin_resnet_model/checkpoint_gsn_latest.pt'
dataset       = 'proposed'
arch          = 'ResNet'
ori_source    = 'rv'          # única opção disponível neste dataset
use_init_rotor= True           # default — ver seção 2
lr            = 2e-5           # bem menor que o treino original, pra fine-tuning
batch_size    = 128
train_list    = 'IMUNet_dataset/list_train_ft.txt'   # 78 sequências (split de list_train.txt)
val_list      = 'IMUNet_dataset/list_val_ft.txt'     # 12 sequências (split de list_train.txt)
                                                       # list_test.txt (36) fica de fora, intocado
epochs        = 130            # checkpoint estava na época 100 → +30 épocas de fine-tuning
```

Comando:

```bash
cd IMUNet/RONIN_torch
python3 finetune_driver.py 2>&1 | tee finetune_train.log
```

Resultado: 30 épocas, ~50-60s cada. Loss de treino caiu de 0.75 (baseline, época 100) pra ~0.035.
Melhor checkpoint (por val loss) na **época 126**, val loss **0.0434** — convergência estável,
sem sinais de overfitting. Checkpoints salvos em
`IMUNet/RONIN_torch/Train_out/ResNet/imunet_dataset_ft/checkpoints/` (`checkpoint_best.pt` e
`checkpoint_latest.pt`).

---

## 5. Teste / avaliação

Script: `IMUNet/RONIN_torch/test_driver.py` (mesmo esquema do fine-tuning, agora chamando
`test_sequence()`). Rodamos **dois** cenários, com o mesmo `model_path` (`checkpoint_best.pt`)
e o mesmo `test_list = IMUNet_dataset/list_test.txt` (36 sequências nunca usadas em treino nem
validação), variando só `use_init_rotor`:

```bash
cd IMUNet/RONIN_torch
python3 test_driver.py 2>&1 | tee test_eval.log
```

### Cenário A — `use_init_rotor=True` ("oráculo", `Test_out/imunet_dataset_ft/`)

Mesmo alinhamento ao frame do ARCore usado no treino. **Não reflete a inferência real** (o
celular nunca tem esse dado), serve só de teto de performance / comparação com benchmarks
do estilo RONIN/IMUNet publicados.

**ATE médio 5.55m, RTE médio 4.24m.**

### Cenário B — `use_init_rotor=False` (simula a inferência real no celular, `Test_out/imunet_dataset_ft_no_init_rotor/`)

**ATE médio 48.05m, RTE médio 42.72m** — quase **9x pior**, e de forma uniforme nas 36
sequências (não são só alguns outliers).

Olhando os plots desse cenário (ex.: `Indoor_Subject_4_Tango_1_gsn.png`), o padrão é sempre o
mesmo: a trajetória predita tem uma forma parecida com o ground truth, mas **rotacionada/
deslocada** — o mesmo sintoma "forma certa, orientação errada" que o modelo IMUNet original
tinha quando treinado *sem* `init_rotor` (ver conversa). A diferença é que agora a rede *foi*
treinada com `init_rotor`; o problema é que ela não generalizou o suficiente pra tolerar a
ausência dele na inferência — o `RandomHoriRotate` (que em teoria ensina essa invariância)
não teve treino suficiente pra "pegar" de verdade: só 30 épocas em 78 sequências, muito menos
dado e muito menos épocas que o treino original do RONIN (100 épocas, dataset bem maior e mais
diverso), que é o que provavelmente explica por que o RONIN original funciona bem sem
`init_rotor` no celular e este fine-tuning específico não (ainda).

### ⚠️ Atualização: o ATE/RTE daqui não alinham rotação antes de medir

`compute_absolute_trajectory_error`/`compute_relative_trajectory_error` (`metric.py:13-58`) são
RMSE **bruta** entre posições/deslocamentos — sem o passo de alinhamento rígido (rotação +
translação) que a definição original de ATE (Sturm et al., citada no próprio docstring) inclui.
Isso significa que os 48m do cenário sem `init_rotor` incluem, de forma indistinguível, tanto
erro de **forma** da trajetória quanto um simples offset de **heading** (rotação) por sequência.

Recalculando o ATE por sequência com um alinhamento ótimo de rotação+translação (Procrustes 2D)
antes de medir o erro: **ATE médio cai de ~68m (bruto, nesta convenção de métrica) pra ~3.9m
(alinhado)** — comparável ou até melhor que o cenário oráculo. A forma da trajetória está sim
bem prevista; o que está "errado" é majoritariamente um offset de heading por sequência.

Detalhe notável: o ângulo de rotação necessário pra alinhar fica concentrado perto de **-90°**
na maioria das sequências, não distribuído aleatoriamente — o que sugere uma convenção de eixo
diferente entre `rv` (referenciado ao norte magnético) e o frame do ARCore, ou um viés do
protocolo de coleta (caminhadas começando em direções parecidas), mais do que "a rede não
aprendeu invariância de heading". Vale investigar a causa exata depois.

### Implicação prática

Se a aplicação tiver **qualquer** referência de heading absoluto uma vez por sessão (bússola,
ponto conhecido, GPS no início), a forma da trajetória reconstruída deve ficar boa mesmo sem
`init_rotor`. Sem nenhuma âncora externa de heading, a ambiguidade de rotação por sessão
continua sendo um problema em aberto — não é (só) falta de treino, como a análise inicial
sugeria.

Saída em cada `Test_out/imunet_dataset_ft*/`:
- `<sequencia>_gsn.png` — plot da trajetória (predito vs groundtruth) + erro de vx/vy, por sequência
- `<sequencia>_gsn.npy`, `<sequencia>pos_cum_error.npy`, `<sequencia>pos_error.npy` — dados brutos dos plots
- `model.onnx` — modelo exportado (mesma arquitetura que o modelo RONIN já embarcado, mas ver ressalva acima antes de embarcar)

---

## 6. Pendências / próximos passos

1. **Prioridade** — investigar a causa do offset de heading de ~-90° concentrado na maioria das
   sequências (seção 5): checar se é convenção de eixo (`rv` referenciado ao norte magnético vs.
   frame do ARCore) ou viés do protocolo de coleta. Se for convenção de eixo, pode ser uma
   correção fixa simples em vez de precisar de mais treino.
2. Decidir se a aplicação final tem alguma forma de âncora de heading (bússola, ponto conhecido)
   pra resolver a ambiguidade de rotação por sessão — se sim, a forma da trajetória (ATE
   alinhado ~3.9m) já está em bom estado.
3. Estender `read_data_s10.py` pra ler o arquivo bruto do `game_rv` e escrever as colunas
   `game_rv_*` no `processed/data.csv` (seção 1).
4. Rodar `read_data_s10.py` no seu dataset de 20h.
5. Fine-tuning encadeado: `--continue_from checkpoint_best.pt` (o desta sessão) + seu dataset +
   `--ori_source game_rv` — testa a segunda hipótese (ruído de magnetômetro do `rv`) que não deu
   pra testar aqui. Ao rodar isso, testar **sempre** os dois cenários (com e sem `init_rotor`) e,
   se o ATE bruto sem `init_rotor` parecer ruim, recalcular com alinhamento de rotação antes de
   concluir algo (igual fizemos aqui) — não repetir o erro de comparar ATE bruto direto.
6. Converter `model.onnx` pra TFLite e validar no celular.
