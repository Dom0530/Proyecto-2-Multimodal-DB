from ..tables.table import VarLenTable, Block, FixedTable
import re
import nltk
import atexit
import os
import json
import math
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

STATE_FILE = "bsb_state.json"
GLOBAL_PTR_I = 0  
GLOBAL_PTR_J = 0 
PL_ORDER = 20
class SPIMI:
    
    def __init__(self, table : VarLenTable , table_field):
        self.table = table
        self.table_field = table_field

        nltk.download('stopwords')

        #global GLOBAL_PTR_I, GLOBAL_PTR_J
        #GLOBAL_PTR_I, GLOBAL_PTR_J = load_state()

        self.posting_table = FixedTable('plt.bin', f'{PL_ORDER*2}i')
        self.directory = FixedTable('directory.bin', '30si')
        self.temp_directory = FixedTable('temp_directory.bin', '30si')
        #self.BSB_index_construction()

        #atexit.register(save_state)

    

    def BSB_index_construction(self):
        global GLOBAL_PTR_I 
        doc_id = 0
        b = []
        while True:
            try:
                current_record = self.table.read(doc_id)
            except:
                break
            
            current_doc = current_record[self.table_field]
            token_stream = self.preprocess(current_doc)
            self.spimi_invert(token_stream, doc_id)

            doc_id += 1
            b.append(GLOBAL_PTR_I - 1)
        
        self.merge_blocks(b, GLOBAL_PTR_I)

    def preprocess(self, text):
        text = text.lower()
        tokens = re.split(r'\W+', text)
        stop_words = set(stopwords.words('english'))
        tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]
        stemmer = SnowballStemmer('english')
        tokens_raiz = [stemmer.stem(t) for t in tokens_filtrados]

        return tokens_raiz
    
    def spimi_invert(self, token_stream, doc_id):
        k = 0
        dictionary = {}
        while k < len(token_stream):
            token = token_stream[k]
            if token not in dictionary:
                posting_list = []
                dictionary[token] = posting_list
            else:             
                posting_list = dictionary[token]
            
            if len(posting_list) == PL_ORDER: 
                posting_list = [posting_list,[]] 
            self.insert_in_posting_list(posting_list, doc_id)
            k += 1
        sorted_terms = dict(sorted(dictionary.items()))
        self.write_to_disk(sorted_terms)
        

    def write_to_disk(self, sorted_terms):

        for term, pl in sorted_terms.items():
            global GLOBAL_PTR_I, GLOBAL_PTR_J
            i, j = GLOBAL_PTR_I, GLOBAL_PTR_J 
            self.temp_directory.write(i, to_bytes(term, 30), j)
            self.posting_table.write(j, *pad_with_nulls(tuple_to_list(pl), PL_ORDER*2))
            GLOBAL_PTR_I += 1
            GLOBAL_PTR_J += 1

        #print('se escribe lo siguiente en el disco')
        #for term, pl in sorted_terms.items():
            #print(f'{term} ->  {pl}')

    def insert_in_posting_list(self, posting_list, doc_id):
        for idx, (doc, freq) in enumerate(posting_list):
            if doc == doc_id:
                posting_list[idx] = (doc, freq + 1)
                return
        posting_list.append((doc_id, 1))


    def merge_blocks(self, b, n):
        global GLOBAL_PTR_I
        GLOBAL_PTR_I = 0
        i = [0]
        t = ['']*(len(b))
        ptr = [0]*(len(b))
        valids = [True] * (len(b))
        

        for b_i in b[:-1]:
            i.append(b_i + 1)
       
        for j in range(len(t)):
            record = self.temp_directory.read(i[j])
            t[j] = record[1]
            ptr[j] = record[2]
        
        for _ in range(n):
            if not True in valids:
                return
            idx = min((x for x in range(len(t)) if valids[x]), key=lambda i: str(t[i]))
            duplicates = [x for x in range(len(t)) if valids[x] and t[x] == t[idx]]

            if len(duplicates) > 1 and (idx in duplicates):
                pls = []
                new_ptr = ptr[duplicates[0]]
                for k in duplicates:
                    ptr[k] = new_ptr
                    pls.append(list_to_tuple(self.posting_table.read(i[k])[1:-1]))
                #print(pls)
                new_pl = [elemento for sublista in pls for elemento in sublista]

                self.directory.write(GLOBAL_PTR_I, t[idx], new_ptr)
                GLOBAL_PTR_I += 1
                self.posting_table.update(new_ptr, *pad_with_nulls(tuple_to_list(new_pl), PL_ORDER*2))

                for k in duplicates:
                    if i[k] + 1 > b[k]:
                        valids[k] = False
                    else:
                        i[k] += 1
                        record = self.temp_directory.read(i[k])
                        t[k] = record[1]
                        ptr[k] = record[2]
                
                continue
                
            self.directory.write(GLOBAL_PTR_I, t[idx], ptr[idx])
            GLOBAL_PTR_I += 1

            if i[idx] + 1 > b[idx]:
                valids[idx] = False
            else:
                i[idx] += 1
                record = self.temp_directory.read(i[idx])
                t[idx] = record[1]
                ptr[idx] = record[2]
    
    def get_posting_list(self, term):
        return self.bin_search(term, GLOBAL_PTR_I)
    
    def bin_search(self, term, n):
        lo = 0
        hi = n - 1
        while lo <= hi:
            m = (lo + hi) // 2
            record = self.directory.read(m)

            if record[1].decode('utf-8').rstrip('\x00') == term:
                pl = self.posting_table.read(record[2])[1:-1]
                return list_to_tuple(pl)
            elif record[1].decode('utf-8').rstrip('\x00') > term:
                hi = m - 1
            elif record[1].decode('utf-8').rstrip('\x00') < term:
                lo = m + 1 
            
        return None
    
    def make_bow(self, tokens):
        bow = defaultdict(int)
        for token in tokens:
            bow[token] += 1
        return dict(bow)
    
    def knn(self, query, k):
        score = {}
        query = self.preprocess(query)
        query = self.make_bow(query)
        N = self.table.count_records()
        norm_q = 0
        tf_idf_query = {}

        for term, tf_query in query.items():

            posting_list = self.get_posting_list(term)
            idf = math.log(1 + (N/len(posting_list)))
            tf_idf_que = math.log(1 + tf_query) * idf
            tf_idf_query[term] = tf_idf_que
            norm_q += tf_idf_que**2
            
        
        norms_d = {}
        for term, tf_idf_que in tf_idf_query.items():

            posting_list = self.get_posting_list(term)
            idf = math.log(1 + N / len(posting_list))
            for doc, tf_doc in posting_list:
                tf_idf_doc = math.log(1 + tf_doc) * idf
                score[doc] = score.get(doc, 0) + tf_idf_doc * tf_idf_que
                norms_d[doc] = norms_d.get(doc, 0) + tf_idf_doc**2

        for doc in score:
            norm_d = math.sqrt(norms_d[doc])
            norm_q_sqrt = math.sqrt(norm_q)
            score[doc] = score[doc] / (norm_d*norm_q_sqrt) if norm_d > 0 and norm_q > 0 else 0
       
        return sorted(score.items(), key=lambda x: x[1], reverse=True)[:k]
    

def list_to_tuple(list_):
    result = []
    for i in range(0,len(list_) - 1, 2):
        if list_[i] == -1:
            break
        result.append((list_[i], list_[i+1]))
    return result

def tuple_to_list(tuple_):
    return [item for tup in tuple_ for item in tup]

def to_bytes(s, length):
    return s.encode('utf-8').ljust(length, b'\x00')[:length]

def pad_with_nulls(arr, length):
    return arr + [-1] * (length - len(arr))

def save_state(i, j):
    with open(STATE_FILE, 'w') as f:
        json.dump({"i": i, "j": j}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            return state["i"], state["j"]
    return 0, 0  # default values


"""
doc = 'In a quiet valley where the flowers bloomed forever, a little bear named Lilo built a bed made of petals—petals so soft that even the stars wanted to sleep in them. Every night, Lilo would count the petals, whisper to the petals, and thank the petals for the dreams they brought. One night, a glowing butterfly landed on her nose and gave her a dream made entirely of flying petals that carried her gently across the sky. Not far from that valley, in a meadow where the grass hummed sweet songs, a kitten named Momo curled up inside a bell-shaped flower. The flower rocked her like a cradle, and every petal around her sang softly—softly enough to make the moon smile. Momo dreamed of flowers, flowers with wings, flowers that danced, and flowers that whispered stories just like this one.'
doc2 = 'In a forest where the trees swayed like waves, a small owl named Nino built a nest of feathers—feathers he found, feathers he borrowed, and feathers the wind left behind. Nino loved feathers so much that he wore one behind his ear, slept on a pile of them, and even dreamed in feathers. One night, a silver feather floated down from the sky, and when Nino touched it, he dreamed of flying through clouds made of feathers and stars. Not far away, under a hill shaped like a pillow, lived a hedgehog named Lila, who collected tiny bells. Bells for her window, bells for her slippers, and bells that chimed when she breathed. One evening, a bell rang all by itself, and when Lila listened, she heard laughter—laughter from the stars, laughter from the moon, and laughter from the dream waiting just beyond sleep.'
doc3 = 'In the heart of the meadow, a gentle wind carried the scent of wildflowers through the morning mist. A young fox, golden as the sun, chased dandelion seeds as if they were stars falling from the sky. Each leap felt like a dance, each breath a song of freedom.'
doc4 = 'Beneath the old willow tree, a girl with braided hair whispered secrets to the stream. The water listened patiently, rippling softly with every word. Somewhere in those gentle waves, the stories of her dreams floated away to the sea.'
doc5 = 'A tiny cabin stood where the forest kissed the mountain, its roof blanketed in moss and memories. Inside, a candle flickered beside a journal filled with poems about owls, snow, and the sound of silence. Time passed slowly, like honey dripping from a spoon.'
doc6 = 'On the edge of the lake, a boy named Arlo built boats from leaves and sent them sailing across the water. He imagined each one carried a message for the moon. As night fell, stars began to mirror his dreams on the gentle waves.'
doc7 = 'In a village where every house had a blue door, a woman painted birds on her windowpanes. She believed that if she painted enough wings, spring would return faster. And when the snow melted, the trees seemed to grow feathers.'
doc8 = 'A soft rain tapped the roof of the greenhouse where vines whispered to sleeping seeds. A black cat curled beneath a table of herbs, dreaming of summer. Outside, puddles collected wishes from passing clouds.'
doc9 =  'High on the hill, a windmill turned slowly, creaking with each gust as if it were sighing. Below, sheep grazed in silence, their wool glowing in the golden dusk. A child watched it all from a swing, her toes brushing the tips of the tall grass.'
table = VarLenTable('docs.txt', 0, 2, [int, str], sep='#')
table.append(0, doc)
table.append(1, doc2)
table.append(2, doc3)
table.append(3, doc4)
table.append(4, doc5)
table.append(5, doc6)
table.append(6, doc7)
table.append(7, doc8)
table.append(8, doc9)
bsb = Bsb(table, 1)



#for i in range(GLOBAL_PTR_I):
    #record = bsb.directory.read(i)
    #print(str(record[1].decode('utf-8')))
    #print(bsb.posting_table.read(record[2]))

print('######## consulta ###################')

print(bsb.knn('Who dreams of flying?', 4))
"""