"""
Texto do documento da entrega. scripts/09_gerar_documento.py lê este arquivo, troca os campos
entre chaves pelos números medidos (resultados/teste/metricas.json e dados/dataset/resumo.json)
e gera o PDF. Assim o texto e as tabelas nunca discordam dos resultados.

Marcação: linha em branco separa parágrafos; linhas "1. " ou "- " viram listas; `código`;
**negrito**; [rótulo](url). Uma tupla ("figura", caminho, legenda) insere uma figura.
"""

TITULO = "Boca de Lobo: detecção em tempo real do estado de bocas de lobo com YOLOv8"
SUBTITULO = "Componente Visão do MVP · modelo yolov8n-bocadelobo-v2"

INTEGRANTES = [
    ("Açussena Macedo Mautone", "552568"),
    ("Felipe Heilmann Marques", "551026"),
    ("Felipe Voidela Toledo", "98595"),
    ("Carlos Eduardo Caramante Ribeiro", "552159"),
    ("Ian Cancian Nachtergaele", "98387"),
]

LINKS = ("Repositório: [github.com/FelipeVoidela/GS2-2026-Applied-Computer-Vision]"
         "(https://github.com/FelipeVoidela/GS2-2026-Applied-Computer-Vision) · "
         "Vídeo de demonstração: [resultados/demo/demo_ronda.mp4]"
         "(https://github.com/FelipeVoidela/GS2-2026-Applied-Computer-Vision/raw/main/resultados/demo/demo_ronda.mp4)"
         ", no repositório (o link baixa o arquivo)")

# ----------------------------------------------------------------------------------- 1. Resumo
RESUMO = """
Todo verão, bairros baixos de São Paulo alagam porque as bocas de lobo estão cobertas de folhas e
lixo, e ninguém sabe quais antes do temporal. No Boca de Lobo, um robô inspeciona esses pontos, e
o sistema cruza o estado de cada um com a previsão de chuva e as áreas de risco para avisar a
equipe de limpeza e os moradores. Este documento descreve o componente de visão: um detector que
classifica cada ponto como bueiro livre, parcialmente obstruído, entupido ou lixo na sarjeta.

Treinamos de forma customizada a arquitetura YOLOv8n, partindo de pesos COCO. O conjunto reúne
{total_fotos} fotos reais de licença livre do Wikimedia Commons, com {total_caixas} caixas
anotadas por pré-anotação com Grounding DINO e revisão visual. Como bueiros obstruídos são raros
nas fotos públicas, o treino recebeu {sinteticas} imagens sintéticas, com recortes reais de
lixo colados em fotos reais de bueiros; validação e teste usam só fotos reais.

No teste ({fotos_teste} fotos), o modelo localiza bem a boca de lobo: na classe bueiro livre o
mAP@50 foi {map50_livre}, e a leitura do ponto acertou
{acertos_ponto} de {fotos_ponto} fotos. A inferência leva {ms} ms por quadro em CPU, cerca de
{fps} quadros por segundo: tempo real. As classes de obstrução não são confiáveis: o
modelo reconheceu {obstr_certos} dos {fotos_obstr_teste} bueiros obstruídos do teste, e o mAP@50 médio das quatro
classes ficou em {map50}. O próximo passo é fotografar
bueiros obstruídos na região piloto e retreinar com o mesmo pipeline, cuja saída segue o
contrato do produto.
"""

# ------------------------------------------------------------------------------- 2. Introdução
INTRODUCAO = [
    """
Alagamento urbano é um problema de drenagem tanto quanto de chuva. A boca de lobo é a entrada da
rede de drenagem na rua; quando folhas, sacolas e embalagens cobrem a grelha ou a abertura na
guia, a água fica na superfície, e uma chuva que a rede comportaria vira rua alagada, carro
perdido e casa invadida. A manutenção hoje é reativa: a prefeitura limpa depois que o morador
reclama (em São Paulo, pelo SP156), em geral depois que o ponto já alagou.

O produto da equipe, o **Boca de Lobo**, inverte essa ordem. Antes de cada temporal, um robô
percorre o bairro e inspeciona as bocas de lobo; quando a previsão indica chuva forte, o sistema
cruza o estado de cada ponto com as áreas de risco e avisa quem precisa agir: a equipe de limpeza,
com uma rota priorizada, e os moradores, que passam a saber onde não passar. O cidadão atendido é
o morador de região que alaga todo verão, principalmente em áreas baixas e periféricas.

Cada disciplina da Global Solution constrói um componente do mesmo sistema. O de **visão** é o que
enxerga o problema: transforma a imagem da câmera do robô em uma informação que o resto do produto
consegue usar. O robô também mede a profundidade livre da sarjeta com ultrassom (Maker Lab), mas o
sensor não distingue uma grelha coberta de folhas de um desnível do asfalto; a câmera, sim. As
duas leituras seguem juntas, na mesma inspeção, para o pipeline de dados (Big Data), que alimenta
o mapa público e o assistente do morador (GenAI).
""",
    ("figura", "docs/pipeline.png",
     "Fluxo do componente de visão, da foto ao produto. As etapas em azul preparam os dados, a "
     "laranja é o treino e as verdes rodam no robô."),
    """
O problema de visão é uma **detecção de objetos** com quatro classes, as mesmas que o pipeline de
dados da equipe aceita no campo `visao.classe`:

- **Bueiro livre** (`livre`): abertura desobstruída, com no máximo sujeira leve.
- **Parcialmente obstruído** (`parcialmente_obstruido`): folhas, lixo ou terra cobrindo parte da abertura, entre um quarto e três quartos.
- **Bueiro entupido** (`entupido`): abertura quase toda coberta ou tomada por lama.
- **Lixo na sarjeta** (`lixo_sarjeta`): lixo ou folhas acumulados na sarjeta, fora do bueiro, que a próxima chuva leva para dentro dele.
""",
    ("figura", "docs/classes.jpg", "Exemplos reais de cada classe, recortados do conjunto de dados."),
]

# --------------------------------------------------------------------------------- 3. Objetivo
OBJETIVO = """
Treinar de forma customizada um detector YOLOv8 que, na imagem da câmera do robô, localize bocas
de lobo e lixo na sarjeta em tempo real e classifique cada ponto como livre, parcialmente
obstruído ou entupido, entregando essa leitura ao pipeline de dados, que alerta moradores e as
equipes de limpeza.
"""

# ------------------------------------------------------------------------------ 4. Ferramentas
FERRAMENTAS = [
    """
**YOLOv8 (Ultralytics).** YOLO (You Only Look Once) é uma família de detectores de um estágio: a
rede olha a imagem inteira uma única vez e devolve, de uma só passada, as caixas e as classes dos
objetos. O YOLOv8 tem três partes: a espinha dorsal (backbone), uma rede convolucional com blocos
C2f que extrai características em três escalas; o pescoço (neck), que combina essas escalas para
que objetos pequenos e grandes sejam vistos com o mesmo detalhe; e a cabeça de detecção, que é
desacoplada (um ramo prevê a caixa, outro a classe) e livre de âncoras, ou seja, prevê diretamente
a distância do ponto até as quatro bordas da caixa. Usamos a versão **nano (YOLOv8n)**, com cerca
de 3 milhões de parâmetros e {pesos_mb} MB de pesos: é a que cabe no computador embarcado do robô
e roda em tempo real sem GPU.

**Transferência de aprendizado.** Em vez de treinar do zero, partimos dos pesos do YOLOv8n
pré-treinados no COCO (80 classes do cotidiano). As primeiras camadas já sabem detectar bordas,
texturas e formas; o treino ajusta a rede inteira ao nosso domínio e troca a cabeça de 80 classes
por uma de 4. É o que torna possível treinar com poucas centenas de imagens.

**PyTorch.** Biblioteca de aprendizado profundo sobre a qual o Ultralytics roda: calcula os
gradientes e executa o treino e a inferência. Usamos a versão de CPU.

**OpenCV.** Biblioteca de visão computacional usada em tudo que mexe com imagem fora da rede: ler
câmera e vídeo, redimensionar, desenhar caixas e rótulos na tela, montar as folhas de revisão e
gerar as imagens sintéticas.

**Grounding DINO (Hugging Face Transformers).** Detector de vocabulário aberto: recebe uma frase
(por exemplo, "storm drain grate") e localiza na imagem o que a frase descreve, sem ter sido
treinado para aquela classe. Usamos apenas para **pré-anotar**: ele propõe as caixas, e a decisão
sobre a classe e sobre quais caixas valem é da revisão. Testamos antes o YOLO-World, que quase não
reconheceu grelhas de bueiro, e por isso foi descartado.

**Wikimedia Commons.** Acervo de mídia de licença livre. Todas as fotos reais do conjunto vêm
dele, pela API pública, com autor e licença registrados em `dados/fontes.csv`.

**Métricas.** Precisão é a fração das detecções que estão certas; revocação é a fração dos objetos
reais que foram encontrados. O mAP@50 resume as duas em todas as classes, contando como acerto a
caixa prevista que se sobrepõe à anotada em pelo menos 50% (IoU); o mAP@50-95 repete a conta
com exigências de sobreposição de 50% a 95% e mede também a qualidade do encaixe da caixa.
""",
]

# ------------------------------------------------------------------------------ 5. Metodologia
METODOLOGIA = [
    """
Cada passo é um script numerado do repositório; rodar os scripts em ordem reproduz o trabalho.

**Passo 1. Coleta das fotos (`scripts/01_coletar_imagens.py`).** O script percorre as categorias
de bocas de lobo, sarjetas e grelhas do Wikimedia Commons e faz buscas por texto, guarda só fotos
com licença livre e baixa a versão de 1280 px. O Commons limita o volume de download de clientes
sem identificação; no prazo da entrega foram baixadas {fotos_baixadas} fotos. Não são fotos tiradas
pela equipe: essa é a principal limitação dos dados, retomada na conclusão.

**Passo 2. Pré-anotação (`scripts/02_pre_anotar.py`).** O Grounding DINO propõe caixas para dois
grupos, "bueiro" e "lixo". Caixas pequenas demais, grandes demais ou repetidas são filtradas.

**Passo 3. Revisão visual (`scripts/03_folhas_revisao.py` e `03b_folhas_manuais.py`).** Cada caixa
proposta vira um recorte numerado em uma folha de contato, e o revisor decide: uma das quatro
classes ou descartar. Foram revisadas {caixas_revisadas} caixas e aceitas {caixas_aceitas}; a
maioria foi descartada, porque as categorias do Commons misturam bocas de lobo com tampas de poço
de visita, galerias, valas e canais. As decisões ficam em `dados/anotacao/revisao.csv`. Fotos sem
nenhuma caixa aceita foram conferidas por amostragem em folhas com régua; caixas que faltavam
foram registradas à mão ({caixas_manuais} no total).
""",
    ("figura", "docs/folha_revisao.jpg",
     "Parte de uma folha de revisão. Cada recorte mostra a caixa proposta, o identificador e o "
     "grupo sugerido pelo pré-anotador (B = bueiro, L = lixo)."),
    """
**Passo 4. Montagem do conjunto (`scripts/04_montar_dataset.py`).** As decisões viram rótulos no
formato YOLO. A divisão em treino, validação e teste é feita por grupo de fotos: imagens do mesmo
autor e da mesma série ficam juntas, para o teste não conter uma foto quase idêntica a uma do
treino. Ela é estratificada pela classe mais rara de cada foto e usa semente fixa.

**Passo 4b. Fotos sintéticas no treino (`scripts/04b_gerar_sinteticos.py`).** Bueiro livre é comum
nas fotos públicas; bueiro entupido é raro, porque pouca gente fotografa o problema. Para o modelo
não aprender só a classe fácil, usamos aumento por copiar e colar: recortes reais de folhas e lixo
(das caixas de lixo na sarjeta do próprio treino) são colados em fotos reais de bueiros livres,
com borda irregular, giro, escala e brilho ajustados. São três tipos de imagem: lixo sobre a
grelha cobrindo de 30% a 65% (parcialmente obstruído); lixo sobre a grelha cobrindo mais de 82%
(entupido); e lixo ao lado do bueiro, caso em que a grelha continua livre e o recorte ganha uma
caixa de lixo na sarjeta. O terceiro tipo foi acrescentado na segunda rodada de treino (seção 6),
junto com a repetição das poucas fotos reais de bueiro obstruído. Tudo isso entra **só no
treino**. Validação e teste têm apenas fotos reais, sem repetição, então as métricas não são
infladas pelas montagens.
""",
    "TABELA_DATASET",
    "TABELA_SINTETICAS",
    ("figura", "docs/sinteticos.jpg",
     "Exemplos das imagens sintéticas do treino: lixo e folhas reais colados em fotos reais. Caixa "
     "amarela: parcialmente obstruído. Vermelha: entupido. Verde: livre. Roxa: lixo na sarjeta."),
    """
**Passo 5. Treinamento (`scripts/05_treinar.py`).** YOLOv8n com pesos do COCO, imagens de 640 px,
lote de 16, até 60 épocas com parada antecipada (25 épocas sem melhora na validação), otimizador
escolhido automaticamente pelo Ultralytics e semente fixa. O treino rodou {epocas_rodadas} épocas
em CPU e os pesos guardados são os da época {melhor_epoca}. O aumento de dados em tempo de treino
reflete o que a câmera do robô vai ver: giros de até 15 graus, espelhamento horizontal e vertical
(visto de cima, o bueiro não tem "lado certo"), variação de cor, saturação e brilho, escala e
mosaico. Ficam guardados os pesos da época de maior mAP na validação.

**Passo 6. Avaliação (`scripts/06_avaliar.py`).** No conjunto de teste medimos: a detecção
(precisão, revocação e mAP por classe); a **leitura do ponto**, que é o que o produto usa (para
cada foto, a classe que o robô enviaria comparada com a anotada); e a velocidade por quadro.

**Passo 7. Tempo real e integração (`detectar.py`).** O programa lê a câmera, um vídeo ou fotos,
roda o modelo quadro a quadro e mostra as caixas, o FPS e o estado do ponto (livre, atenção ou
crítico). O estado exibido é o mais votado nos últimos quadros, para o alerta não piscar quando
uma detecção falha por um instante. Com `--json`, cada leitura sai no formato do contrato com o
pipeline de dados: modelo, classe, confiança e `foto_id`. O `foto_id` é um hash da imagem: a foto
não sai do robô, o que evita transmitir rostos e placas que apareçam na cena (LGPD).

**Passo 8. Demonstração (`scripts/07_gerar_video_demo.py`).** Como ainda não há filmagem do robô
em campo, o vídeo é uma ronda simulada: cada foto de teste vira alguns segundos de vídeo com
movimento de câmera, e o modelo roda em todos os quadros durante a gravação.
""",
]

# ------------------------------------------------------------------------------- 6. Resultados
RESULTADOS = [
    """
**Detecção no conjunto de teste.** A tabela traz as métricas por classe nas {fotos_teste} fotos de
teste, que o modelo não viu no treino nem na escolha dos pesos.
""",
    "TABELA_DETECCAO",
    """
Precisão 1,000 com revocação 0,000 quer dizer que o modelo não previu a classe nenhuma vez no
teste: não há predição para errar, e também nenhum acerto.

A leitura da tabela tem duas metades. **O modelo aprendeu a encontrar a boca de lobo:** na classe
bueiro livre, a única com volume de fotos reais, o mAP@50 é {map50_livre} e a revocação é
{rev_livre}, com precisão de {prec_livre}: ele acha a maior parte das grelhas do teste, de formatos e
países diferentes, e quase nunca aponta uma onde não há.
**O modelo ainda não aprendeu a graduar a obstrução:** as classes parcialmente obstruído e entupido
ficaram perto de zero, e lixo na sarjeta ficou baixa. Por isso a média das quatro classes
({map50}) é bem menor que o resultado da classe livre. Na validação, as classes livre e
parcialmente obstruído repetem o padrão. Lixo na sarjeta foi melhor na validação (mAP@50 de
{map50_lixo_val}) do que no teste ({map50_lixo}); com 4 ou 5 caixas em cada conjunto, essa
diferença não permite conclusão. A classe entupido tem uma única foto real em cada conjunto.

A causa está nos dados. O treino tem {reais_obstr_treino} caixas reais das duas classes de
obstrução, contra {reais_livre_treino} de bueiro livre. Conferimos o que o modelo aprendeu
(`scripts/06b_conferir_treino.py`): ele acerta {montagens_obstr_certas} das {montagens_obstr_total}
montagens de bueiro obstruído e {reais_obstr_treino_certas} das {reais_obstr_treino_total} fotos
reais obstruídas do próprio treino, mas {obstr_certos} das {fotos_obstr_teste} fotos obstruídas do
teste. Ou seja, decorou as montagens e os poucos exemplos reais, sem aprender a obstrução como ela
aparece em fotos novas, em que costuma ser sutil: folhas secas da cor do asfalto, terra, grama.
Com lixo na sarjeta o problema é parecido: {lixo_treino_maior_autor} das {lixo_treino_total}
caixas reais do treino são do mesmo fotógrafo, na mesma rua. O conjunto de teste também é pequeno
({fotos_obstr_teste} fotos com bueiro obstruído), então os números dessas classes dizem "ainda não
funciona", mas não medem com precisão o quanto falta.
""",
    ("figura", "resultados/treino/results.png",
     "Curvas do treinamento: perdas de treino e de validação e métricas de validação por época."),
    """
**Leitura do ponto.** É a medida mais próxima do uso real: uma foto, uma resposta. O modelo
acertou {acertos_ponto} de {fotos_ponto} fotos ({acuracia_ponto}), com confiança mínima de {conf}.
Linhas: classe anotada. Colunas: classe que o robô enviaria.
""",
    "TABELA_PONTO",
    """
O acerto geral é puxado pela classe livre ({acertos_livre} de {fotos_livre} fotos certas). O erro
que mais importa para o cidadão é o bueiro com problema que passa sem alerta, porque é ele que
alaga a rua: dos {com_problema} pontos de teste com problema (obstruídos ou com lixo na sarjeta),
{nao_alertado} foram lidos como livres ou não foram vistos. Dos {alertados} alertas, {alertas_lixo}
vieram de pontos com lixo na sarjeta e {obstr_alertados} de bueiros obstruídos. No sentido oposto,
{livres_como_problema} bueiros livres foram lidos como problema. Na versão atual, portanto, a visão
serve para **localizar e contar as bocas de lobo** de uma ronda e confirmar as que estão limpas,
mas não substitui o sensor de ultrassom do robô na decisão de alertar. No produto, as duas
leituras seguem juntas, e a prioridade de limpeza não depende só da câmera.
""",
    ("figura", "docs/exemplos_teste.jpg",
     "Predições em fotos de teste. Caixa branca: anotação. Caixa colorida: predição do modelo. "
     "A faixa inferior mostra o estado do ponto."),
    """
**Duas rodadas de treino.** O modelo entregue é o da segunda rodada. Na primeira, as imagens
sintéticas eram só montagens de lixo sobre a grelha, e o modelo aprendeu a reconhecer a colagem em
vez da obstrução. Na segunda mudamos três coisas no treino, sem tocar em validação e teste: lixo
colado também ao lado de bueiros que continuam livres, para a marca da colagem deixar de denunciar
a classe; colagem com borda mais nítida, sem o borrão do giro; e repetição das fotos reais de
bueiro obstruído. O critério para trocar de modelo foi fixado antes de medir a segunda rodada e
usa só a **validação**: alertar pelo menos um ponto com problema a mais, sem ler mais de dois
bueiros livres como problema e sem perder acerto nos livres. A tabela mostra as duas rodadas; o
teste aparece para registro.
""",
    "TABELA_RODADAS",
    """
A segunda rodada cumpriu o critério ({val_alertados} de {val_com_problema} pontos com problema
alertados na validação, contra nenhum na primeira) e melhorou a localização: mais leituras certas
na validação e lixo na sarjeta reconhecido em parte. Mas o ganho é de uma ou duas fotos em
conjuntos de 26, e o principal não mudou: nenhuma das duas rodadas reconhece bueiro obstruído em
foto nova. Ajuste de treino não resolveu; o que falta é dado real.
""",
    """
**Tempo real.** Medido foto a foto no conjunto de teste, na máquina de desenvolvimento, sem GPU.
""",
    "TABELA_VELOCIDADE",
    """
A {fps} quadros por segundo o modelo acompanha uma câmera comum com folga. No vídeo de
demonstração, com detecção, desenho do painel e codificação do vídeo ao mesmo tempo, o sistema
manteve {fps_demo} quadros por segundo em {pontos_demo} pontos de inspeção. A ronda terminou com
{demo_livre} pontos lidos como livres, {demo_atencao} em atenção, {demo_critico} críticos e
{demo_sem} sem leitura (o modelo não viu boca de lobo em pelo menos um terço dos quadros). Os
pontos do vídeo incluem os bueiros obstruídos do teste, e ele mostra os erros descritos acima do
mesmo jeito que mostra os acertos.
""",
    ("figura", "docs/demo_quadro1.jpg",
     "Quadro do vídeo de demonstração: detecção à esquerda e, à direita, a leitura enviada ao "
     "pipeline de dados e o resumo da ronda."),
]

# -------------------------------------------------------------------------------- 7. Conclusão
CONCLUSAO = """
O componente de visão está construído de ponta a ponta e é reproduzível: coleta com registro de
licença, anotação assistida com revisão, conjunto de dados dividido sem vazamento entre treino e
teste, treinamento customizado do YOLOv8n, avaliação, detecção em tempo real e saída no contrato
que o pipeline de dados do produto já consome. O modelo roda a {fps} quadros por segundo em CPU,
com {pesos_mb} MB de pesos, o que o torna viável no computador embarcado do robô, e encontra a
boca de lobo com mAP@50 de {map50_livre} em fotos de vários países que nunca viu.

O objetivo foi atingido em parte. A detecção em tempo real e a integração funcionam; a
classificação do grau de obstrução, não. Preferimos registrar esse resultado como ele é a
apresentar só a média favorável: um sistema de alerta que deixa passar bueiro entupido é pior do
que um que avisa que ainda não sabe.

O que aprendemos e o que vem a seguir:

- **Dado real da classe difícil não tem substituto.** As imagens sintéticas não ensinaram a obstrução real, nem na primeira rodada nem depois dos ajustes da segunda. O próximo passo é fotografar bocas de lobo obstruídas na região piloto (zona leste de São Paulo), com a câmera e o ângulo do robô, e retreinar: o pipeline aceita as fotos novas sem mudança de código.
- **Boca de lobo de guia.** O conjunto é dominado por grelhas, comuns na Europa e nos Estados Unidos. Em São Paulo predomina a abertura na guia, quase ausente nas fotos públicas. As fotos da equipe corrigem também esse viés geográfico.
- **Conjunto de teste maior.** Com {fotos_teste} fotos, as métricas das classes raras são indicativas. A meta é ter pelo menos 50 exemplos reais de cada classe no teste.
- **Uso combinado com o ultrassom.** Enquanto a visão não gradua a obstrução, o estado do ponto no produto deve vir do sensor de profundidade, com a câmera confirmando que há uma boca de lobo no quadro.
"""

# ------------------------------------------------------------------------------- Referências
REFERENCIAS = [
    """
- Jocher, G.; Chaurasia, A.; Qiu, J. Ultralytics YOLOv8. 2023. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Liu, S. et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024.
- Ghiasi, G. et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR 2021.
- Lin, T.-Y. et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
- Wikimedia Commons. [commons.wikimedia.org](https://commons.wikimedia.org)

**Licenças das fotos.** As fotos do conjunto de dados são de terceiros, publicadas no Wikimedia
Commons sob licença livre. O autor, a licença e a página de cada uma estão em `dados/fontes.csv`,
no repositório. Distribuição das licenças entre as fotos usadas:
""",
    "TABELA_LICENCAS",
]
