"""
Texto do documento da entrega. scripts/09_gerar_documento.py lê este arquivo, troca os campos
entre chaves pelos números medidos (resultados/teste/metricas.json e dados/dataset/resumo.json)
e gera o PDF. Assim o texto e as tabelas nunca discordam dos resultados.

Marcação: linha em branco separa parágrafos; linhas "1. " ou "- " viram listas; `código`;
**negrito**; [rótulo](url). Uma tupla ("figura", caminho, legenda) insere uma figura.
"""

TITULO = "Boca de Lobo: detecção em tempo real do estado de bocas de lobo com YOLOv8"
SUBTITULO = "Componente Visão do MVP · YOLOv8n (localiza) + graduador de obstrução (classifica)"

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

O sistema tem duas etapas, ambas com treinamento customizado. O YOLOv8n, partindo de pesos COCO,
localiza bocas de lobo e lixo em tempo real. O graduador, camadas lineares sobre os codificadores
CLIP e DINOv2, decide no recorte se o bueiro está obstruído e, quando nada é detectado, se há
material acumulado no quadro. O conjunto reúne {total_fotos} fotos reais de licença livre, do
Wikimedia Commons e do Flickr, com {total_caixas} caixas anotadas; validação e teste usam só fotos
reais.

No teste ({fotos_teste} fotos), o sistema alertou {s_alert} dos {s_prob} pontos com problema, contra
{y_alert} do YOLO sozinho, e acertou {s_certas} leituras, contra {y_certas}. Dos {s_nobs} bueiros
obstruídos, {s_obs_alert} geraram alerta, {s_obs} deles lidos como obstruídos; antes, nenhum.
Em contrapartida, {s_livres_prob} dos {s_livres} bueiros livres viraram alarme falso. O YOLO roda a {fps} quadros por
segundo em CPU, e o sistema completo, em vídeo, a {fps_cad}. Resta graduar a obstrução: faltam fotos reais de
bueiros obstruídos, raras nos acervos públicos e esgotadas nas nossas buscas.
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
O problema de visão é uma **detecção de objetos** em tempo real com quatro classes, as mesmas que o
pipeline de dados da equipe aceita no campo `visao.classe`:

- **Bueiro livre** (`livre`): abertura desobstruída, com no máximo sujeira leve.
- **Parcialmente obstruído** (`parcialmente_obstruido`): folhas, lixo ou terra cobrindo parte da abertura, entre um quarto e três quartos.
- **Bueiro entupido** (`entupido`): abertura quase toda coberta ou tomada por lama.
- **Lixo na sarjeta** (`lixo_sarjeta`): lixo ou folhas acumulados na sarjeta, fora do bueiro, que a próxima chuva leva para dentro dele.
""",
    ("figura", "docs/classes.jpg", "Exemplos reais de cada classe, recortados do conjunto de dados."),
]

# --------------------------------------------------------------------------------- 3. Objetivo
OBJETIVO = """
Treinar de forma customizada um detector YOLOv8 e um graduador que, na câmera do robô, localizem
bocas de lobo em tempo real e classifiquem cada ponto como livre, parcialmente obstruído, entupido
ou lixo na sarjeta, entregando essa leitura ao pipeline de dados, que alerta moradores e as equipes de
limpeza.
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

**Wikimedia Commons e Openverse.** Acervos de mídia de licença livre. O Commons tem categorias de
bocas de lobo e grelhas. O Openverse é um buscador de fotos de licença aberta, principalmente do
Flickr, e foi usado nas buscas dirigidas a bueiro obstruído. O autor e a licença de cada foto ficam
registrados em `dados/fontes.csv`.

**CLIP (OpenAI) e DINOv2 (Meta).** São codificadores de imagem: redes já treinadas que transformam
uma imagem em um vetor de números que resume o que há nela. O CLIP (ViT-B/32) aprendeu com centenas
de milhões de pares de imagem e legenda, e por isso seu vetor carrega significado ("folhas", "lixo",
"grelha"). O DINOv2 (ViT-S/14) aprendeu só com imagens, sem legenda, e é forte em textura e
estrutura, como as frestas de uma grelha. Usamos os dois **congelados**, isto é, sem alterar seus
pesos: eles formam a base da segunda etapa do sistema, o graduador.

**Camada linear (regressão logística).** É o que treinamos em cima dos codificadores: uma única
camada que recebe o vetor da imagem e devolve uma pontuação. Com duas dezenas de exemplos da classe
rara, ajustar algumas centenas de pesos sobre uma rede que já enxerga bem generaliza melhor do que
ajustar os 3 milhões de pesos do YOLO. É também transferência de aprendizado, em sua forma mais
enxuta.

**Métricas.** Precisão é a fração das detecções que estão certas; revocação é a fração dos objetos
reais que foram encontrados. O mAP@50 resume as duas em todas as classes, contando como acerto a
caixa prevista que se sobrepõe à anotada em pelo menos 50% (IoU); o mAP@50-95 repete a conta
com exigências de sobreposição de 50% a 95% e mede também a qualidade do encaixe da caixa. A AUC,
usada no graduador, é a chance de um exemplo com problema receber pontuação maior que um sem
problema: 0,5 é sorteio e 1,0 é separação perfeita.
""",
]

# ------------------------------------------------------------------------------ 5. Metodologia
METODOLOGIA = [
    """
Cada passo é um script numerado do repositório; rodar os scripts em ordem reproduz o trabalho.

**Passo 1. Coleta das fotos (`scripts/01_coletar_imagens.py`, `01b_coletar_obstruidos.py` e
`01c_achar_repetidas.py`).** O primeiro script percorre as categorias de bocas de lobo, sarjetas e
grelhas do Wikimedia Commons e faz buscas por texto. O segundo, acrescentado na terceira rodada de
treino, faz buscas dirigidas a bueiro obstruído ("blocked drain", "storm drain leaves", "bueiro
entupido") no Openverse, que indexa o Flickr. Só entram fotos com licença que permite redistribuir
e modificar. Foram baixadas {fotos_baixadas} fotos ({fotos_commons} do Commons e {fotos_flickr} do
Flickr). Como a mesma foto aparece nas duas fontes, o terceiro script compara uma assinatura de
cada imagem e marca as cópias: {repetidas} fotos repetidas ficaram de fora, uma delas cópia de uma
foto de teste. Na quarta rodada refizemos as buscas com outras palavras e em outros idiomas: das
{fotos_rodada4} fotos novas, só duas mostram um bueiro obstruído, e o conjunto de dados não mudou.
Os acervos públicos estão esgotados para essa classe. Não são fotos tiradas pela equipe: essa é a
principal limitação dos dados, retomada na conclusão.

**Passo 2. Pré-anotação (`scripts/02_pre_anotar.py`).** O Grounding DINO propõe caixas para dois
grupos, "bueiro" e "lixo". Caixas pequenas demais, grandes demais ou repetidas são filtradas.

**Passo 3. Revisão visual (`scripts/03_folhas_revisao.py` e `03b_folhas_manuais.py`).** Cada caixa
proposta vira um recorte numerado em uma folha de contato, e o revisor decide: uma das quatro
classes ou descartar. As buscas por texto trazem muito ruído (tampas de poço de visita, rios,
pessoas), então as fotos do Flickr passaram antes por uma triagem em miniaturas, e só as
aproveitáveis foram pré-anotadas. Foram revisadas {caixas_revisadas} caixas e aceitas
{caixas_aceitas}. As decisões ficam em `dados/anotacao/revisao.csv`. Onde o detector não propôs
caixa, o que é comum quando a grelha está coberta, a caixa foi desenhada à mão em folhas com régua
({caixas_manuais} no total).
""",
    ("figura", "docs/folha_revisao.jpg",
     "Parte de uma folha de revisão. Cada recorte mostra a caixa proposta, o identificador e o "
     "grupo sugerido pelo pré-anotador (B = bueiro, L = lixo)."),
    """
**Passo 4. Montagem do conjunto (`scripts/04_montar_dataset.py`).** As decisões viram rótulos no
formato YOLO. A divisão em treino, validação e teste é feita por grupo de fotos: imagens do mesmo
autor e da mesma série ficam juntas (nas fotos do Flickr, todas as do mesmo autor), para o teste
não conter uma foto quase idêntica a uma do treino. Ela é estratificada pela classe mais rara de
cada foto, usa semente fixa e fica registrada em `dados/anotacao/divisao.csv`: quando o conjunto
cresceu, as fotos antigas continuaram onde estavam e só as novas foram sorteadas, o que permite
comparar as rodadas de treino nas mesmas fotos.

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
escolhido automaticamente pelo Ultralytics e semente fixa. O treino do modelo entregue (segunda rodada) rodou
{epocas_rodadas} épocas em CPU, e os pesos guardados são os da época {melhor_epoca}; o da terceira
rodada parou sozinho na época {epocas_rodada3}. O aumento de dados em tempo de treino
reflete o que a câmera do robô vai ver: giros de até 15 graus, espelhamento horizontal e vertical
(visto de cima, o bueiro não tem "lado certo"), variação de cor, saturação e brilho, escala e
mosaico. Ficam guardados os pesos da época de maior mAP na validação.

**Passo 5b. Treinamento do graduador (`scripts/11_treinar_graduador.py`).** É a segunda etapa do
sistema, acrescentada na quarta rodada, e tem duas cabeças. A cabeça de **recorte** recebe o recorte
de uma caixa de boca de lobo e responde se ela está obstruída. A cabeça de **quadro** só entra
quando o YOLO não detecta nada: recebe o quadro inteiro e responde se há problema no ponto (lixo,
folhas, grelha encoberta), porque uma boca de lobo totalmente coberta não tem grelha para detectar.
Cada cabeça é uma camada linear sobre o CLIP e outra sobre o DINOv2; as duas pontuações são
padronizadas e a média é a pontuação final. O treino usa só as fotos reais de treino (as montagens
não ajudaram), cada exemplo com pequenas variações de enquadramento e espelhado. A regularização é
escolhida por validação cruzada em cinco dobras, separadas por autor da foto, para fotos da mesma
série nunca ficarem dos dois lados. O limiar é fixado no próprio treino: a pontuação que apenas
{g_quantil} dos exemplos sem problema ultrapassam, medida fora da dobra. Depois de tudo escolhido só
com o treino e medido na validação, as camadas finais são reajustadas com treino e validação
juntos; o teste continua fora.

**Passo 6. Avaliação (`scripts/06_avaliar.py`).** No conjunto de teste medimos: a detecção do YOLO
(precisão, revocação e mAP por classe); a **leitura do ponto**, que é o que o produto usa (para
cada foto, a classe que o robô enviaria comparada com a anotada), com o sistema completo e só com
o YOLO; e a velocidade por quadro. As decisões entre modelos usam só a validação
(`--so-validacao`), e a validação do sistema é medida com as camadas treinadas só no treino.

**Passo 7. Tempo real e integração (`detectar.py`).** O programa lê a câmera, um vídeo ou fotos e
mostra as caixas, o FPS e o estado do ponto (livre, atenção ou crítico). O YOLO roda em todos os
quadros; o graduador, mais pesado, roda a cada {cadencia} quadros, e nos quadros intermediários cada
caixa herda a pontuação da caixa que mais se sobrepõe a ela. O estado exibido é o mais votado nos
últimos quadros, para o alerta não piscar quando uma detecção falha por um instante. Com `--json`,
cada leitura sai no formato do contrato com o pipeline de dados: modelo, classe, confiança e
`foto_id`. O `foto_id` é um hash da imagem: a foto não sai do robô, o que evita transmitir rostos e
placas que apareçam na cena (LGPD). Se a segunda etapa não puder ser carregada, o programa avisa e
segue só com o YOLO.

**Passo 8. Demonstração (`scripts/07_gerar_video_demo.py`).** Como ainda não há filmagem do robô
em campo, o vídeo é uma ronda simulada: cada foto de teste vira alguns segundos de vídeo com
movimento de câmera, e o sistema roda durante a gravação, como rodaria na câmera.
""",
]

# ------------------------------------------------------------------------------- 6. Resultados
RESULTADOS = [
    """
**Primeira etapa: o YOLO localiza.** A tabela traz as métricas de detecção do YOLO, por classe, nas
{fotos_teste} fotos de teste, que nenhuma das duas etapas viu no treino nem na escolha dos pesos.
""",
    "TABELA_DETECCAO",
    """
A leitura da tabela tem duas metades. **O YOLO aprendeu a encontrar a boca de lobo:** na classe
bueiro livre, a única com volume de fotos reais, o mAP@50 é {map50_livre} e a precisão é de
{prec_livre}: quando ele aponta uma grelha, quase sempre há uma. A revocação é de {rev_livre}: ele
ainda deixa passar parte das grelhas do teste, que tem formatos, países e fotógrafos que o treino
não viu. **O YOLO não aprendeu a graduar a obstrução:** nas classes parcialmente obstruído e
entupido a revocação é zero, e lixo na sarjeta ficou baixa. Por isso a média das quatro classes
({map50}) é bem menor que o resultado da classe livre.

A causa está nos dados. O YOLO entregue foi treinado com 12 caixas reais das duas classes de
obstrução, contra 92 de bueiro livre. Conferimos o que ele aprendeu
(`scripts/06b_conferir_treino.py`): acerta {c2_montagens_obstr_certas} das
{c2_montagens_obstr_total} montagens de bueiro obstruído e {c2_reais_obstr_treino_certas} das
{c2_reais_obstr_treino_total} fotos reais obstruídas do próprio treino, mas, sozinho, {y_obs} das
{s_nobs} fotos obstruídas do teste. Ou seja, decorou as montagens e os poucos exemplos reais, sem
aprender a obstrução como ela aparece em fotos novas, em que costuma ser sutil: folhas secas da cor
do asfalto, terra, grama. O bueiro totalmente coberto traz outra dificuldade: sem grelha à vista,
não há o que detectar. São esses dois problemas que a segunda etapa ataca.
""",
    ("figura", "resultados/treino/results.png",
     "Curvas do treinamento do YOLO: perdas de treino e de validação e métricas de validação por época."),
    """
**Segunda etapa: o graduador classifica.** A tabela resume o treino das duas cabeças. "Fora da
dobra" quer dizer que cada exemplo do treino foi pontuado por camadas que não o viram (validação
cruzada por autor da foto). O limiar deixa passar {g_quantil} dos exemplos sem problema do treino.
""",
    "TABELA_GRADUADOR",
    """
Nos recortes anotados, a cabeça de recorte separa o bueiro obstruído do livre com AUC de
{g_recorte_auc} no treino e {g_recorte_auc_val} na validação: reconhece {g_recorte_oof_pos} dos
{g_recorte_pos} bueiros obstruídos do treino e {g_recorte_val_pos} dos {g_recorte_val_npos} da
validação. A cabeça de quadro separa o ponto com problema do ponto livre com AUC de {g_quadro_auc}
e {g_quadro_auc_val}. Os dois codificadores se completam: o DINOv2 foi melhor no treino e o CLIP na
validação, e a média dos dois foi a opção mais estável. Vale o aviso de sempre: com tão poucos
exemplos da classe rara, uma ou duas fotos mudam esses números.

**Leitura do ponto.** É a medida mais próxima do uso real: uma foto, uma resposta, com confiança
mínima de {conf} para as caixas do YOLO. A tabela compara o YOLO sozinho com o sistema de duas etapas.
""",
    "TABELA_ETAPAS",
    """
O erro que mais importa para o cidadão é o ponto com problema que passa sem alerta, porque é ele
que alaga a rua. Dos {s_prob} pontos de teste com problema (bueiros obstruídos ou lixo na sarjeta),
o YOLO sozinho alertava {y_alert}; o sistema alerta {s_alert}, e {s_naoalert} ainda passam. Dos
{s_nobs} bueiros obstruídos, nenhum gerava alerta; agora {s_obs_alert} geram. No sentido oposto,
{s_livres_prob} dos {s_livres} bueiros livres são lidos como problema (eram {y_livres_prob}). O
acerto geral sobe de {y_certas} para {s_certas} em {s_fotos} fotos ({s_acuracia}). A tabela
seguinte detalha as leituras do sistema no teste. Linhas: classe anotada. Colunas: classe que o
robô enviaria.
""",
    "TABELA_PONTO",
    """
Duas ressalvas, para o número não dizer mais do que mede. **Alertar não é graduar:** dos
{s_obs_alert} bueiros obstruídos que geram alerta, só {s_obs} são lidos como obstruídos; os outros
recebem a leitura "lixo na sarjeta", que na cabeça de quadro significa material acumulado sem grelha
à vista. O status é o mesmo (atenção), e o ponto entra na rota de limpeza, mas o sistema ainda não
distingue o bueiro coberto do lixo solto, nem o parcialmente obstruído do entupido. **A localização
continua sendo o gargalo:** {s_livres_nada} bueiros livres do teste não são encontrados pelo YOLO, e
a segunda etapa só gradua o que a primeira localiza.
""",
    ("figura", "docs/exemplos_teste.jpg",
     "Leituras do sistema em fotos de teste. Caixa branca: anotação. Caixa colorida: caixa do YOLO com a "
     "classe dada pelo graduador. A faixa inferior mostra o estado do ponto."),
    """
**Quatro rodadas.** Na primeira, as imagens sintéticas eram só montagens de lixo sobre a grelha, e
o YOLO aprendeu a reconhecer a colagem em vez da obstrução. Na segunda mudamos três coisas no
treino: lixo colado também ao lado de bueiros que continuam livres, para a marca da colagem deixar
de denunciar a classe; colagem com borda mais nítida; e repetição das fotos reais de bueiro
obstruído. Na terceira atacamos a falta de dado real: as buscas dirigidas do passo 1 levaram o
conjunto de 162 para {total_fotos} fotos reais, e as caixas reais de bueiro obstruído no treino, de
12 para {reais_obstr_treino}. Na quarta, sem mais fotos para buscar, mudamos a arquitetura: o YOLO
da segunda rodada passou a só localizar, e o graduador a classificar.

Em cada troca, o critério foi fixado antes e usa só a **validação**. A primeira tabela mostra as
três primeiras rodadas nas mesmas 26 fotos de validação e 26 de teste das duas primeiras. A segunda
mostra as rodadas na validação e no teste ampliados ({fotos_val} e {fotos_teste} fotos). O mAP é
sempre o do YOLO; por isso é igual na segunda e na quarta rodadas, que usam o mesmo YOLO.
""",
    "TABELA_RODADAS",
    "TABELA_RODADAS_NOVO",
    """
A terceira rodada não cumpriu o critério dela (reconhecer pelo menos dois bueiros obstruídos a mais
que a segunda, ler como problema no máximo 10% dos bueiros livres e não perder mais de cinco pontos
percentuais de acerto nos livres): na validação ampliada reconheceu {r3v_obs} dos {v_nobs} bueiros
obstruídos (a segunda, {r2v_obs}), e repetiu o padrão de acertar no treino o que erra fora dele
({c3_reais_obstr_treino_certas} das {c3_reais_obstr_treino_total} fotos reais obstruídas do treino).
Os pesos dela ficam no repositório para conferência.

O experimento que fechou a terceira rodada (`scripts/10_experimento_clip.py`) apontou o caminho da
quarta: mesmo sem treino, o CLIP separava bem bueiro obstruído de livre nos recortes anotados (AUC
de {clip_auc_treino} no treino e {clip_auc_val} na validação), mas de ponta a ponta a leitura não
mudava ({clip_obs} de {clip_nobs} bueiros obstruídos), porque o limiar era rígido e parte dos
bueiros obstruídos nem era detectada. A quarta rodada respondeu aos dois pontos: camadas treinadas
sobre dois codificadores, com limiar tirado da validação cruzada, e a cabeça de quadro para quando
não há caixa.

Para a quarta rodada o critério mudou de alvo, e dizemos isso com clareza. Na validação, o sistema
tinha de alertar pelo menos dois pontos com problema a mais que o YOLO sozinho, ler como problema
no máximo 10% dos bueiros livres e não perder mais de cinco pontos percentuais de acerto nos
livres. Cumpriu: alertou {sv_alert} dos {sv_prob} pontos com problema (o YOLO sozinho,
{yv_alert}), leu como problema {sv_livres_prob} dos {sv_livres} bueiros livres e acertou
{sv_livres_certos} deles (antes, {yv_livres_certos}). Pelo critério antigo, de bueiros lidos
**como obstruídos**, não cumpriria: foram {sv_obs} de {sv_nobs}, um a mais que antes. Só depois
dessa decisão o teste foi medido, uma vez.

Dois caminhos testados na quarta rodada não entraram. Um YOLO com as três classes de boca de lobo
unificadas em uma só, para a obstrução deixar de competir com a localização, localizou na validação
os mesmos 4 dos 6 bueiros obstruídos que o YOLO da segunda rodada. E uma terceira cabeça, para
distinguir nas caixas de lixo o bueiro coberto do lixo solto, separava bem os recortes anotados mas
não mudou nenhuma leitura de ponta a ponta.
""",
    """
**Tempo real.** Medido foto a foto no conjunto de teste, na máquina de desenvolvimento, sem GPU.
""",
    "TABELA_VELOCIDADE",
    """
O YOLO, que roda em todos os quadros, faz {fps} quadros por segundo e acompanha uma câmera comum
com folga. O graduador é mais pesado: com ele em todos os quadros, o sistema cai para {fps_2e}
quadros por segundo. Como o estado de um bueiro não muda de um quadro para o outro, em vídeo o
graduador roda a cada {cadencia} quadros, o que dá, pela conta dos dois tempos, {fps_cad} quadros por segundo. No vídeo de
demonstração, com detecção, graduação, desenho do painel e codificação do vídeo ao mesmo tempo, o
sistema manteve {fps_demo} quadros por segundo em {pontos_demo} pontos de inspeção. A ronda terminou
com {demo_livre} pontos lidos como livres, {demo_atencao} em atenção, {demo_critico} críticos e
{demo_sem} sem leitura. Dos {demo_prob} pontos com problema anotado, {demo_prob_alert} geraram
alerta; dos {demo_livres} livres, {demo_livres_ok} foram lidos como livres. Com a câmera em
movimento as leituras oscilam mais que na foto parada, e o grau da obstrução aparece trocado em
vários pontos: o vídeo mostra os erros do mesmo jeito que mostra os acertos.
""",
    ("figura", "docs/demo_quadro1.jpg",
     "Quadro do vídeo de demonstração: detecção à esquerda e, à direita, a leitura enviada ao "
     "pipeline de dados e o resumo da ronda."),
]

# -------------------------------------------------------------------------------- 7. Conclusão
CONCLUSAO = """
O componente de visão está construído de ponta a ponta e é reproduzível: coleta com registro de
licença, anotação assistida com revisão, conjunto de dados dividido sem vazamento entre treino e
teste, treinamento customizado em duas etapas, avaliação, detecção em tempo real e saída no
contrato que o pipeline de dados do produto já consome. O YOLO roda a {fps} quadros por segundo em
CPU, com {pesos_mb} MB de pesos, e encontra a boca de lobo com mAP@50 de {map50_livre} em fotos de
vários países que nunca viu.

O objetivo foi atingido no que mais pesa para o cidadão e segue aberto em um ponto. O sistema
passou a avisar: {s_alert} dos {s_prob} pontos com problema do teste geram alerta, contra
{y_alert} com o YOLO sozinho, com {s_livres_prob} alarmes falsos em {s_livres} bueiros livres. O que
ele ainda não faz bem é **graduar**: só {s_obs} dos {s_nobs} bueiros obstruídos são lidos como
obstruídos, e a diferença entre parcialmente obstruído e entupido não é confiável. Preferimos
registrar as duas metades do resultado a apresentar só a favorável.

O que aprendemos e o que vem a seguir:

- **Com pouco dado, treinar menos pesos sobre uma rede que já enxerga.** Três rodadas ajustando o YOLO inteiro, com montagens e com mais fotos, não ensinaram a obstrução. Camadas lineares sobre codificadores pré-treinados, com os mesmos dados, ensinaram o bastante para o alerta funcionar.
- **Quando não há o que detectar, olhar o quadro.** Uma boca de lobo totalmente coberta não tem grelha. A cabeça de quadro responde à pergunta que resta: há material acumulado neste ponto?
- **Dado real da classe difícil ainda é o limite, e as fotos públicas se esgotaram.** Em {fotos_baixadas} fotos públicas baixadas, pouco mais de trinta mostram um bueiro obstruído, e as buscas refeitas na quarta rodada trouxeram só mais duas. Para graduar a obstrução e para a boca de lobo de guia, comum em São Paulo e quase ausente nos acervos, o caminho é fotografar na região piloto (zona leste), com a câmera e o ângulo do robô: o pipeline aceita as fotos novas sem mudança de código.
- **Medir antes de trocar.** Critério fixado antes, decisão pela validação e teste medido uma vez evitaram adotar um modelo por uma diferença que era ruído, e deixaram claro quando o critério mudou.
- **Conjunto de teste maior.** Com {fotos_teste} fotos e {reais_obstr_teste} caixas de bueiro obstruído, as métricas das classes raras são indicativas. A meta é ter pelo menos 50 exemplos reais de cada classe no teste.
- **Uso combinado com o ultrassom.** A visão já serve para decidir se o ponto merece atenção; o grau (atenção ou crítico) deve continuar sendo confirmado pelo sensor de profundidade do robô, e quem decide mandar a equipe é uma pessoa.
"""

# ------------------------------------------------------------------------------- Referências
REFERENCIAS = [
    """
- Jocher, G.; Chaurasia, A.; Qiu, J. Ultralytics YOLOv8. 2023. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Liu, S. et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024.
- Ghiasi, G. et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR 2021.
- Lin, T.-Y. et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
- Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024.
- Wikimedia Commons. [commons.wikimedia.org](https://commons.wikimedia.org)
- Openverse. [openverse.org](https://openverse.org)

**Licenças das fotos.** As fotos do conjunto de dados são de terceiros, publicadas no Wikimedia
Commons e no Flickr sob licença livre. O autor, a licença e a página de cada uma estão em
`dados/fontes.csv`, no repositório. Distribuição das licenças entre as fotos usadas:
""",
    "TABELA_LICENCAS",
]
