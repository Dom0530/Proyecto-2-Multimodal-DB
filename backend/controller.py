import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.parser import full_statement
from index.spimi import SPIMI
from ..config import TYPE_OF_KNN, INVERTED_INDEX_PATH, CODEBOOK_PATH, IDF_PATH, FEATURES_PATH, DATASET_ROOT
from index.invert_idx import BoAWRetriever

class Index:
    def __init__(self, type, varlen_table, field_num):
        global INVERTED_INDEX_PATH, CODEBOOK_PATH, IDF_PATH, FEATURES_PATH, DATASET_ROOT
        if type == 'SPIMI':
            self.index = SPIMI(varlen_table, field_num)
            self.index.index_load_from_disk()
            if self.index == None:
                self.index.BSB_index_construction()

        elif type == 'IVEC':
            self.index = BoAWRetriever()

            self.index.load_model_from_disk(CODEBOOK_PATH, IDF_PATH, FEATURES_PATH, INVERTED_INDEX_PATH)
            if self.index == None:
                self.index.fit_and_export(dataset_root=DATASET_ROOT, output_csv="features.csv")
        else:
            raise Exception(f'type <{type}>: not valid')
        

    def make_query(self, sql_query, limit, offset):

        try:
            parse = full_statement.parseString(sql_query, parseAll=True)[0]
        except Exception as e:
            raise Exception(f"Error de sintaxis «{sql_query}»: {e}")
        
        # Busqueda
        if parse['tipo'] == 'SELECT':
            tabla = parse['tabla']
            columnas = parse['columnas']
            where = parse['where']
            limit_query = parse['limit']
        
            index = self.index

            if where == None:
                try:
                    data = index.load(limit, columnas)
                    return ('OK', data, offset+1)
                except Exception as e:
                    print(f"Error en el indice «{where[0]}» : {e}")
            
            elif where[1] == '@@':
                try:
                    data = index.knn(where[2], min(limit_query, limit))
                    return ('OK', data, offset+1)
                except Exception as e:
                    print(f"Error en la busqueda KNN : {e}")

            elif where[1] == '<->':
                try:
                    global TYPE_OF_KNN
                    if TYPE_OF_KNN == 'lineal':
                        data = index.query(where[2], min(limit_query, limit))
                    else:
                        data = index.query_inverted_knn(where[2], min(limit_query, limit))
                    return ('OK', data, offset+1)
                except Exception as e:
                    print(f"Error en la busqueda KNN : {e}")


  
    

