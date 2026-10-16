# versión 1.2

import bisect
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
nltk.download('punkt_tab', quiet=True)  # descarga automática si no está disponible
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel

## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")

    if modelname == "SBERT": return SentenceBertEmbeddingModel()
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia

        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria

    NOTA IMPORTANTE: No se usan sets de Python en ningún método. Para las operaciones
    de pertenencia sobre posting lists (que son listas ordenadas) se utiliza el módulo
    bisect de la biblioteca estándar, que proporciona búsqueda binaria O(log n).
    Para la deduplicación de tokens se usan diccionarios (cuyas claves son únicas).
    """

    # Campo del artículo que se indexa
    DEFAULT_FIELD = 'all'
    # Número máximo de resultados a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "positional", "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesarias pero
        puedes añadir más variables si las necesitas.

        Estructura del índice invertido según el modo:
          - Sin posicionales: self.index[termino] = [artid_0, artid_1, ...]
          - Con posicionales: self.index[termino] = [(artid_0, [pos_0, pos_1, ...]),
                                                      (artid_1, [pos_0, pos_1, ...]), ...]
        En ambos casos las listas de artids están ordenadas de forma creciente,
        lo que permite aplicar los algoritmos de merge eficientes.
        """
        # Diccionario de URLs ya procesadas; clave = URL, valor = None.
        # Se usa dict en lugar de set para no depender del tipo set de Python.
        self.urls = {}

        # Índice invertido: clave = término, valor = posting list (ver docstring)
        self.index = {}

        # Mapeo docid → ruta del fichero JSON
        self.docs = {}

        # Mapeo artid → (docid, numero_de_linea_en_fichero)
        # Permite recuperar el artículo original sin escanear todo el fichero.
        self.articles = {}

        # Expresión regular para tokenizar: divide por cualquier carácter no alfanumérico
        self.tokenizer = re.compile(r"\W+")

        # Si es False, se muestran solo los primeros SHOW_MAX resultados
        self.show_all = False

        # Modo posicional: se establece en index_dir
        self.positional = False

        # PARA LA AMPLIACION SEMÁNTICA
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None
        self.model = None
        self.MAX_EMBEDDINGS = 200  # Máximo de embeddings extraídos por consulta


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

    def primera_funcion(self):
        """
        NECESARIO PARA LA VERSION MINIMA

        Primera función a implementar, se llama desde el main y se utiliza
        para configurar el indexador.
        Se pueden añadir argumentos a esta función para configurar el indexador.
        """
        pass


    def set_showall(self, v: bool):
        """
        Cambia el modo de mostrar los resultados.

        input: "v" booleano.
        UTIL PARA TODAS LAS VERSIONES

        Si self.show_all es True se mostrarán todos los resultados en lugar de
        un máximo de self.SHOW_MAX. No aplicable a la opción -C.
        """
        self.show_all = v


    def set_semantic_threshold(self, v: float):
        """
        Cambia el umbral para la búsqueda semántica.

        input: "v" float con el umbral de distancia.
        UTIL PARA LA AMPLIACIÓN

        Si self.semantic es False el umbral no tendrá efecto.
        """
        self.semantic_threshold = v


    def set_semantic_ranking(self, v: bool):
        """
        Cambia el valor de semantic_ranking.

        input: "v" booleano.
        UTIL PARA LA AMPLIACIÓN

        Si self.semantic_ranking es True se hará una consulta binaria y los
        resultados se rankearán por similitud semántica.
        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################

    def save_info(self, filename: str):
        """Guarda la información del índice en un fichero en formato binario."""
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)


    def load_info(self, filename: str):
        """Carga la información del índice desde un fichero en formato binario."""
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

    def load_semantic_model(self, modelname: str = SEMANTIC_MODEL):
        """
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez.
        """
        if self.model is None:
            print(f"loading {modelname} model ... ", end="", file=sys.stderr)
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)


    def update_chuncks(self, txt: str, artid: int):
        """
        Añade los chuncks (frases) del texto "txt" del artículo "artid" a las
        estructuras de datos semánticas.

        Pasos:
            1 - Extraer las frases de txt usando nltk.sent_tokenize en español.
            2 - Para cada frase, registrar:
                  · self.chuncks      → lista global de todas las frases.
                  · self.chunck_index → lista paralela a chuncks con el artid
                                        al que pertenece cada frase.
                  · self.artid_to_emb → dict {artid: [idx_chunk, ...]} que permite
                                        saber qué posiciones de chuncks pertenecen
                                        a un artículo dado.

        Estas estructuras se usan después en create_kdtree y en las búsquedas
        semánticas para pasar de un índice de chunk a un artid.
        """
        # 1 - Extraer frases del texto del artículo
        frases = nltk.sent_tokenize(txt, language='spanish')

        # 2 - Registrar cada frase en las estructuras globales
        for frase in frases:
            idx_chunk = len(self.chuncks)    # Índice de esta frase en la lista global
            self.chuncks.append(frase)
            self.chunck_index.append(artid)  # Qué artículo contiene esta frase

            # Actualizar el mapa artid → lista de índices de chunks
            if artid not in self.artid_to_emb:
                self.artid_to_emb[artid] = []
            self.artid_to_emb[artid].append(idx_chunk)


    def create_kdtree(self):
        """
        Genera los embeddings de todos los chuncks y construye el KDTree.
        Solo se debe llamar una vez, después de indexar todos los documentos.

        # 1: Llamar a self.model.fit(self.chuncks), que internamente:
        #      a) Genera un embedding por cada chunk (frase).
        #      b) Construye el KDTree sobre esos embeddings.
        # 2: Guardar el KDTree y los embeddings en el SAR_Indexer para que
        #    puedan persistirse con save_info y restaurarse al cargar el índice
        #    sin tener que recalcular todos los embeddings.
        """
        print("Creating kdtree ...", end="")

        # 1 - Ajustar el modelo: genera embeddings y construye el KDTree interno
        self.model.fit(self.chuncks)

        # 2 - Guardar referencias para persistencia con pickle (save_info / load_info)
        self.kdtree = self.model.kdtree
        self.embeddings = self.model.embeddings

        print("done!")


    def solve_semantic_query(self, query: str):
        """
        Resuelve una consulta usando similitud semántica con umbral de distancia.

        Devuelve los artículos cuyos chuncks tienen una distancia al embedding
        de la consulta menor o igual a self.semantic_threshold, ordenados de
        más a menos similares.

        Pasos:
            1 - Cargar el modelo y restaurar el KDTree si el índice se cargó
                desde fichero (en ese caso self.model.kdtree es None pero
                self.kdtree sí tiene el árbol guardado con save_info).
            2 - Consultar el modelo con top_k = MAX_EMBEDDINGS.
                Resultado: lista de (distancia, idx_chunk) ordenada por distancia.
            3 - Filtrar por umbral: conservar solo los que tengan
                distancia <= self.semantic_threshold.
            4 - Si el último resultado recuperado TAMBIÉN supera el umbral
                (puede haber más resultados válidos fuera del top_k actual),
                duplicar top_k y repetir desde 2.
            5 - Salir también si top_k >= total de chuncks (ya los recuperamos todos).
            6 - Convertir los índices de chunk a artids únicos (preservando el
                orden de relevancia semántica).

        return: lista de artids ordenados por similitud semántica.
        """
        self.load_semantic_model()

        # 1 - Restaurar KDTree si el modelo acaba de cargarse (búsqueda sin re-indexar)
        if self.model.kdtree is None and self.kdtree is not None:
            self.model.set_kdtree(self.kdtree)
            self.model.set_embeddings(self.embeddings)

        total_chuncks = len(self.chuncks)
        top_k = self.MAX_EMBEDDINGS

        artids_resultado = []
        artids_vistos = {}  # dict en lugar de set: cumple restricción del enunciado

        while True:
            # 2 - Consultar el modelo con el top_k actual (no superar el total)
            top_k_actual = min(top_k, total_chuncks)
            resultados = self.model.query(query, top_k=top_k_actual)
            # resultados = [(distancia, idx_chunk), ...] ordenado por distancia ascendente

            # 3 - Filtrar por umbral y construir lista de artids únicos
            artids_resultado = []
            artids_vistos = {}
            for distancia, idx_chunk in resultados:
                if distancia <= self.semantic_threshold:
                    artid = self.chunck_index[idx_chunk]
                    if artid not in artids_vistos:
                        artids_vistos[artid] = None
                        artids_resultado.append(artid)

            # 4 y 5 - Decidir si hay que ampliar top_k
            ultimo_dentro_umbral = (
                len(resultados) > 0
                and resultados[-1][0] <= self.semantic_threshold
            )
            if ultimo_dentro_umbral and top_k_actual < total_chuncks:
                # El último resultado recuperado cumple el umbral → puede haber más
                top_k *= 2
            else:
                # O el último no cumple el umbral, o ya recuperamos todos los chuncks
                break

        # 6 - artids_resultado ya está en orden de relevancia semántica
        return artids_resultado


    def semantic_reranking(self, query: str, articles: List[int]):
        """
        Reordena la lista 'articles' (resultado de una búsqueda booleana) según
        su similitud semántica con 'query', del más similar al menos similar.

        Pasos:
            1 - Cargar el modelo y restaurar el KDTree si es necesario.
            2 - Consultar el modelo con top_k = MAX_EMBEDDINGS.
            3 - Convertir los índices de chunk devueltos a artids únicos
                (preservando el orden semántico).
            4 - Comprobar si TODOS los artículos de 'articles' están entre los
                recuperados semánticamente. Si no, duplicar top_k y repetir
                desde 2 (salvo que ya hayamos recuperado todos los chuncks).
            5 - Reordenar 'articles' según el orden semántico:
                  a) Primero los artículos de 'articles' que aparecen en los
                     resultados semánticos (en el orden devuelto por el KDTree).
                  b) Después los artículos de 'articles' que NO aparecieron en
                     los resultados semánticos (sin orden garantizado).

        param:  "query"    cadena de consulta.
                "articles" lista de artids resultado de la búsqueda booleana.
        return: 'articles' reordenado por similitud semántica.
        """
        self.load_semantic_model()

        # 1 - Restaurar KDTree si hace falta (índice cargado desde fichero)
        if self.model.kdtree is None and self.kdtree is not None:
            self.model.set_kdtree(self.kdtree)
            self.model.set_embeddings(self.embeddings)

        total_chuncks = len(self.chuncks)
        top_k = self.MAX_EMBEDDINGS

        # Dict de artids del resultado binario para búsqueda O(1) sin set
        artids_binarios_dict = {}
        for artid in articles:
            artids_binarios_dict[artid] = None

        artids_semanticos_ordenados = []
        artids_semanticos_vistos = {}

        while True:
            # 2 - Consultar el modelo
            top_k_actual = min(top_k, total_chuncks)
            resultados = self.model.query(query, top_k=top_k_actual)

            # 3 - Construir lista ordenada semánticamente de artids únicos
            artids_semanticos_ordenados = []
            artids_semanticos_vistos = {}
            for _distancia, idx_chunk in resultados:
                artid = self.chunck_index[idx_chunk]
                if artid not in artids_semanticos_vistos:
                    artids_semanticos_vistos[artid] = None
                    artids_semanticos_ordenados.append(artid)

            # 4 - Comprobar si todos los artículos binarios están cubiertos
            todos_cubiertos = True
            for artid in articles:
                if artid not in artids_semanticos_vistos:
                    todos_cubiertos = False
                    break

            if todos_cubiertos or top_k_actual >= total_chuncks:
                break  # Ya tenemos suficiente o no hay más chuncks

            top_k *= 2

        # 5 - Reordenar 'articles' según el orden semántico
        resultado_reordenado = []

        # a) Artículos que aparecen en el ranking semántico, en orden de similitud
        for artid in artids_semanticos_ordenados:
            if artid in artids_binarios_dict:
                resultado_reordenado.append(artid)

        # b) Artículos del resultado binario no encontrados semánticamente (al final)
        for artid in articles:
            if artid not in artids_semanticos_vistos:
                resultado_reordenado.append(artid)

        return resultado_reordenado


    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article: Dict) -> bool:
        """
        Comprueba si el artículo ya ha sido indexado consultando su URL.

        Args:
            article (Dict): diccionario con la información de un artículo.

        Returns:
            bool: True si la URL del artículo ya está en self.urls.
        """
        return article['url'] in self.urls


    def index_dir(self, root: str, **args):
        """
        Recorre recursivamente el directorio o fichero "root" e indexa su contenido.
        NECESARIO PARA TODAS LAS VERSIONES

        Los argumentos adicionales "**args" son necesarios para las funcionalidades
        ampliadas:
          - positional (bool): si True, construye un índice posicional.
          - semantic  (bool): si True, construye índices semánticos.
        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()

        file_or_dir = Path(root)

        if file_or_dir.is_file():
            self.index_file(root)
        elif file_or_dir.is_dir():
            for directorio, _, ficheros in os.walk(root):
                for nombre_fichero in sorted(ficheros):
                    if nombre_fichero.endswith('.json'):
                        ruta_completa = os.path.join(directorio, nombre_fichero)
                        self.index_file(ruta_completa)
        else:
            print(f"ERROR: {root} no es un fichero ni un directorio.", file=sys.stderr)
            sys.exit(-1)

        # Si el índice semántico está activo, construir el KDTree una vez que
        # se han procesado todos los artículos y se han extraído todos los chuncks.
        if self.semantic:
            self.create_kdtree()


    def parse_article(self, raw_line: str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una línea del fichero del crawler.

        Args:
            raw_line: una línea del fichero generado por el crawler.

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'.
        """
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(
                subsec['name'] + '\n' + subsec['text'] + '\n'
                for subsec in sec['subsections']
            ) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections')
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)
        return article


    def index_file(self, filename: str):
        """
        Indexa el contenido de un fichero JSON (uno o varios artículos, uno por línea).

        input: "filename" es el nombre de un fichero generado por el Crawler;
               cada línea es un objeto JSON con la información de un artículo de Wikipedia.

        NECESARIO PARA TODAS LAS VERSIONES

        Según el valor de self.positional, el índice almacena:
          - False: self.index[termino] = [artid_0, artid_1, ...]
          - True:  self.index[termino] = [(artid_0, [pos_0, pos_1, ...]),
                                          (artid_1, [pos_0, pos_1, ...]), ...]
        Las listas de artids siempre están en orden creciente porque los artids se
        asignan de forma secuencial al procesar los artículos.
        """
        docid = len(self.docs)
        self.docs[docid] = filename

        for numero_linea, linea in enumerate(open(filename, encoding='utf-8')):
            articulo = self.parse_article(linea)

            # Ignorar artículos duplicados (misma URL ya procesada)
            if self.already_in_index(articulo):
                continue
            self.urls[articulo['url']] = None  # Registrar URL como procesada

            tokens = self.tokenize(articulo[self.DEFAULT_FIELD])

            artid = len(self.articles)
            # Guardamos el número de línea real para poder recuperar el artículo
            self.articles[artid] = (docid, numero_linea)

            if self.positional:
                # Modo posicional: almacenar la posición de cada aparición del token.
                # Usamos un dict local para agrupar posiciones por token antes de
                # actualizar el índice global, evitando búsquedas O(n) repetidas.
                posiciones_por_token = {}
                for posicion, token in enumerate(tokens):
                    if token not in posiciones_por_token:
                        posiciones_por_token[token] = []
                    posiciones_por_token[token].append(posicion)

                for token, lista_posiciones in posiciones_por_token.items():
                    if token not in self.index:
                        self.index[token] = []
                    # Cada entrada es una tupla (artid, lista_de_posiciones_ordenada)
                    self.index[token].append((artid, lista_posiciones))

            else:
                # Modo no posicional: almacenar solo los artids (sin repetición).
                # Se usa un dict local para deduplicar tokens sin usar set().
                tokens_unicos = {}
                for token in tokens:
                    tokens_unicos[token] = None

                for token in tokens_unicos:
                    if token not in self.index:
                        self.index[token] = []
                    self.index[token].append(artid)

            if self.semantic:
                self.update_chuncks(articulo[self.DEFAULT_FIELD], artid)


    def tokenize(self, text: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "text" eliminando símbolos no alfanuméricos y
        dividiéndola por espacios.

        params: 'text': texto a tokenizar.
        return: lista de tokens en minúsculas.
        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadísticas del índice construido.
        """
        total_postings = sum(len(lista) for lista in self.index.values())

        print("=" * 40)
        print(f"Ficheros indexados:   {len(self.docs)}")
        print(f"Artículos indexados:  {len(self.articles)}")
        print(f"Términos distintos:   {len(self.index)}")
        print(f"Postings totales:     {total_postings}")
        print(f"Índice posicional:    {'sí' if self.positional else 'no'}")
        print("=" * 40)


    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query: str, prev: Dict = {}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta booleana y devuelve la posting list resultante.

        Sintaxis soportada:
          - Término simple:        gato
          - Negación:              NOT gato
          - Conjunción (AND):      gato AND perro    (también implícito: gato perro)
          - Frase posicional:      "fin de semana"   (requiere índice posicional)
          - Combinaciones:         "fin de semana" AND NOT viaje

        param:  "query"  cadena con la consulta.
                "prev"   incluido por compatibilidad (no se usa en esta versión).

        return: tupla (posting_list, tokens_de_la_query)
                  posting_list: lista ordenada de artids que satisfacen la consulta.
                  tokens:       lista de tokens tal como se parsearon (útil para debug).
        """
        if query is None or len(query) == 0:
            return [], []

        # --- Paso 1: tokenizar la query preservando frases entre comillas ---
        # Ejemplo: 'NOT "fin de semana" AND viaje'
        #   → ['NOT', '"fin de semana"', 'AND', 'viaje']
        tokens_query = []
        posicion = 0
        while posicion < len(query):
            caracter = query[posicion]
            if caracter == ' ':
                posicion += 1
            elif caracter == '"':
                # Buscar la comilla de cierre
                cierre = query.find('"', posicion + 1)
                if cierre == -1:
                    cierre = len(query) - 1
                tokens_query.append(query[posicion:cierre + 1])
                posicion = cierre + 1
            else:
                # Token normal (hasta el siguiente espacio o comilla)
                fin = posicion
                while fin < len(query) and query[fin] not in (' ', '"'):
                    fin += 1
                tokens_query.append(query[posicion:fin])
                posicion = fin

        # --- Paso 2: evaluar los tokens de izquierda a derecha ---
        # AND es la operación por defecto entre términos consecutivos.
        # NOT invierte la siguiente posting list.
        resultado = None
        negar_siguiente = False

        for token in tokens_query:
            token_mayus = token.upper()

            if token_mayus in ('AND', 'OR'):
                # AND es implícito; OR no está implementado (se ignora)
                continue

            if token_mayus == 'NOT':
                negar_siguiente = True
                continue

            posting = self.get_posting(token)

            if negar_siguiente:
                posting = self.reverse_posting(posting)
                negar_siguiente = False

            if resultado is None:
                resultado = posting
            else:
                resultado = self.and_posting(resultado, posting)

        return (resultado if resultado is not None else []), tokens_query


    def get_posting(self, term: str):
        """
        Devuelve la posting list asociada a un término o frase.
        Para frases entre comillas delega en get_positionals.

        param:  "term": término o frase entrecomillada a recuperar.
        return: lista ordenada de artids.

        NECESARIO PARA TODAS LAS VERSIONES
        """
        # Las frases entre comillas se resuelven con búsqueda posicional
        if term.startswith('"') and term.endswith('"'):
            terminos_frase = self.tokenize(term[1:-1])
            return self.get_positionals(terminos_frase)

        if self.positional:
            # El índice posicional almacena (artid, posiciones): extraer solo artids
            return [artid for artid, _posiciones in self.index.get(term, [])]

        # Índice no posicional: devolver copia para no modificar el índice
        return list(self.index.get(term, []))


    def get_positionals(self, terms: List[str]):
        """
        Devuelve la posting list de los artículos donde los términos aparecen
        de forma consecutiva (en el mismo orden que en "terms").

        NECESARIO PARA LAS BÚSQUEDAS POSICIONALES

        Algoritmo:
          1. Obtener la posting list posicional de cada término.
          2. Para cada artículo en la posting list del primer término:
               a. Comprobar que el artículo aparece en TODOS los demás términos.
               b. Para cada posición p del primer término en ese artículo,
                  verificar que el término k (k=1,2,...) aparece en la posición p+k.
                  La verificación usa búsqueda binaria (bisect) sobre la lista de
                  posiciones (que está ordenada), evitando el uso de set().
          3. Si se encuentra alguna posición válida, añadir el artículo al resultado.

        param:  "terms": lista de términos consecutivos.
        return: lista ordenada de artids donde los términos son consecutivos.
        """
        if not terms:
            return []

        if not self.positional:
            # Sin índice posicional no podemos verificar consecutividad;
            # devolvemos los artículos que contienen todos los términos (AND).
            resultado = None
            for term in terms:
                posting = list(self.index.get(term, []))
                resultado = posting if resultado is None else self.and_posting(resultado, posting)
            return resultado if resultado is not None else []

        # Obtener las posting lists posicionales de cada término.
        # Si algún término no existe en el índice, la intersección es vacía.
        postings_por_termino = []
        for term in terms:
            if term not in self.index:
                return []
            postings_por_termino.append(self.index[term])

        # Construir un diccionario {artid: lista_de_posiciones} por término.
        # Las listas de posiciones ya están ordenadas (se generaron en order).
        # Se usa dict en lugar de set para cumplir la restricción del enunciado.
        dicts_posiciones = []
        for posting in postings_por_termino:
            dict_artid = {}
            for artid, lista_posiciones in posting:
                dict_artid[artid] = lista_posiciones
            dicts_posiciones.append(dict_artid)

        resultado = []

        # Iterar sobre los artículos del primer término como pivote
        for artid, posiciones_primer_termino in postings_por_termino[0]:

            # Paso a): el artículo debe aparecer en todos los términos restantes
            aparece_en_todos = True
            for dict_k in dicts_posiciones[1:]:
                if artid not in dict_k:
                    aparece_en_todos = False
                    break
            if not aparece_en_todos:
                continue

            # Paso b): buscar una posición de anchor tal que pos+k exista en termino k
            encontrado = False
            for posicion_anchor in posiciones_primer_termino:
                consecutivos = True
                for k in range(1, len(terms)):
                    posiciones_k = dicts_posiciones[k][artid]
                    posicion_buscada = posicion_anchor + k
                    # Búsqueda binaria: evita recorrer la lista completa (no usa set)
                    idx = bisect.bisect_left(posiciones_k, posicion_buscada)
                    if idx >= len(posiciones_k) or posiciones_k[idx] != posicion_buscada:
                        consecutivos = False
                        break
                if consecutivos:
                    encontrado = True
                    break

            if encontrado:
                resultado.append(artid)

        return resultado


    def reverse_posting(self, p: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list complementaria: todos los artids del índice
        que NO están en "p".

        El algoritmo es un merge lineal O(n) sobre dos listas ordenadas:
        la lista completa de artids (0, 1, ..., N-1) y la lista "p".
        No usa set() para cumplir la restricción del enunciado.

        param:  "p": posting list de artids a excluir.
        return: lista ordenada de artids con todos los artículos excepto los de "p".
        """
        total_articulos = len(self.articles)
        resultado = []
        idx_p = 0  # Puntero sobre la lista p

        for artid in range(total_articulos):
            # Avanzar el puntero de p hasta el artid actual o más allá
            while idx_p < len(p) and p[idx_p] < artid:
                idx_p += 1
            # Si el artid no está en p, se incluye en el resultado
            if idx_p >= len(p) or p[idx_p] != artid:
                resultado.append(artid)

        return resultado


    def and_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula la intersección de dos posting lists de forma EFICIENTE.
        Algoritmo: merge lineal O(n+m) sobre dos listas ordenadas.
        No necesita set() porque aprovecha el orden de las listas.

        param:  "p1", "p2": posting lists sobre las que calcular la intersección.
        return: posting list con los artids presentes en p1 Y en p2.
        """
        resultado = []
        idx1 = 0
        idx2 = 0

        while idx1 < len(p1) and idx2 < len(p2):
            if p1[idx1] == p2[idx2]:
                resultado.append(p1[idx1])
                idx1 += 1
                idx2 += 1
            elif p1[idx1] < p2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

        return resultado


    def minus_posting(self, p1: list, p2: list):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula la diferencia p1 - p2 de forma EFICIENTE.
        Algoritmo: merge lineal O(n+m) sobre dos listas ordenadas.
        No usa set() para cumplir la restricción del enunciado.

        param:  "p1", "p2": posting lists ordenadas.
        return: posting list con los artids presentes en p1 pero NO en p2.
        """
        resultado = []
        idx2 = 0  # Puntero sobre p2

        for artid in p1:
            # Avanzar el puntero de p2 hasta el artid actual o más allá
            while idx2 < len(p2) and p2[idx2] < artid:
                idx2 += 1
            # Si el artid no está en p2, se incluye en el resultado
            if idx2 >= len(p2) or p2[idx2] != artid:
                resultado.append(artid)

        return resultado


    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql: List[str], verbose: bool = True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql: List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)
        return not errors


    def solve_and_show(self, query: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta, muestra los resultados y devuelve el número de
        artículos recuperados.

        Selecciona automáticamente el modo de búsqueda según la configuración:
          · Si self.semantic y self.semantic_threshold no es None:
              búsqueda semántica pura (filtrado por similitud con umbral).
          · Si self.semantic y self.semantic_ranking es True:
              búsqueda booleana + reordenamiento semántico de resultados.
          · En caso contrario:
              búsqueda booleana clásica.

        Para cada artículo se muestra su posición en el ranking, su título y
        su URL. Si self.show_all es False, se limitan los resultados a SHOW_MAX.

        param:  "query": consulta a resolver.
        return: número de artículos recuperados.
        """
        # --- Seleccionar modo de búsqueda ---
        if self.semantic and self.semantic_threshold is not None:
            # Búsqueda semántica pura: devuelve artids ordenados por similitud
            lista_artids = self.solve_semantic_query(query)

        elif self.semantic and self.semantic_ranking:
            # Búsqueda booleana + reranking semántico
            lista_artids, _ = self.solve_query(query)
            lista_artids = self.semantic_reranking(query, lista_artids)

        else:
            # Búsqueda booleana clásica
            lista_artids, _ = self.solve_query(query)

        num_resultados = len(lista_artids)

        print(f"=== Consulta: {query} ===")
        print(f"Número de resultados: {num_resultados}")

        articulos_a_mostrar = lista_artids if self.show_all else lista_artids[:self.SHOW_MAX]

        for posicion, artid in enumerate(articulos_a_mostrar):
            docid, numero_linea = self.articles[artid]
            ruta_fichero = self.docs[docid]

            # Leer solo la línea necesaria del fichero (no cargamos todo en memoria)
            with open(ruta_fichero, encoding='utf-8') as fh:
                for num_linea_actual, linea in enumerate(fh):
                    if num_linea_actual == numero_linea:
                        articulo = json.loads(linea)
                        titulo = articulo.get('title', '(sin titulo)')
                        url = articulo.get('url', '(sin URL)')
                        try:
                            print(f"  {posicion + 1:3}. {titulo}")
                        except UnicodeEncodeError:
                            # Terminal sin soporte UTF-8: sustituir caracteres problemáticos
                            print(f"  {posicion + 1:3}. {titulo.encode('ascii', errors='replace').decode('ascii')}")
                        print(f"       {url}")
                        break

        return num_resultados
