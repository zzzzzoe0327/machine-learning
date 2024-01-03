# coding=utf-8
import csv
import pickle
import re
from os import listdir
from os.path import isfile, join
import math

def write_liste_csv(liste_ligne_csv, file_name='data/out.csv', delimiter=',', quotechar='`'):
    # 將列表中的行寫入 CSV 檔案
    f = open(file_name, 'w+', newline='', encoding='utf-8')
    writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
    for p in liste_ligne_csv:
        writer.writerow(p)
    f.close()

def save_object(o, object_path):
    # 將物件序列化並儲存至檔案
    pickle.dump(o, open(object_path, 'wb'))

def load_object(obj_path):
    # 從檔案載入並還原序列化的物件
    return pickle.load(open(obj_path, 'rb'))

# 驗證所有列表元素的條件 all(map(is_arabic, city))
# nCk
def c_n_k(n, k):
    """
    使用 Andrew Dalke 提供的快速計算二項式係數的方法。
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

# 階乘
def factorial(x):
    x = int(x)
    result = 1
    while x > 1:
        result = result * x
        x = x - 1
    return result

# 整數部分函數 E()
def partie_entiere(x):
    if x == int(x):
        return x
    elif x >= 0:
        return int(x)
    else:
        return -partie_entiere(-x) - 1

# 歐幾里得算法求最大公因數
def pgcd_iterative(a, b):
    while a % b != 0:
        a, b = b, a % b
    return b

def pgcd_recursive(a, b):
    if a % b == 0:
        return b
    else:
        return pgcd_recursive(b, a % b)

# 最小公倍數
def ppmc(a, b):
    return (a * b) / pgcd_recursive(a, b)

# 判斷是否為質數
def is_premier(n):
    if n == 0 or n == 1:
        return False
    else:
        for i in range(2, int(math.sqrt(n))):
            if n % i == 0:
                return False
        return True

# 分解為質因數
def decompsition_premier(n):
    liste = []
    if is_premier(n) or n == 1 or n == 0:
        liste.append((n, 1))
    else:
        i = 2
        while n // i != 0:
            j = 0
            if n % i == 0:
                while n % i == 0:
                    j += 1
                    n = n // i
                liste.append((i, j))
            else:
                i += 1

    return liste

# 來自 scipy.comb()，但已修改
def c_n_k_scipy(n, k):
    if (k > n) or (n < 0) or (k < 0):
        return 0
    top = n
    val = 1
    while top > (n - k):
        val *= top
        top -= 1
    n = 1
    while n < k + 1:
        val /= n
        n += 1
    return val

def read_text_file(path, with_anti_slash=False):
    # 讀取文本檔案
    f = open(path, "r+", encoding='utf-8')
    data = f.readlines()
    if not with_anti_slash:
        for i in range(len(data)):
            data[i] = re.sub(r"\n", "", data[i]).strip()
    return data

def write_liste_in_file(liste, path='data/out.txt'):
    # 寫入列表至檔案
    f = open(path, 'w+', encoding='utf-8')
    liste = list(map(str, liste))
    for i in range(len(liste)-1):
        liste[i] = str(liste[i]) + "\n" 
    f.writelines(liste)

def strip_and_split(string_in):
    return string_in.strip().split()

def to_upper_file_text(path_source, path_destination):
    # 將檔案中的文本轉換為大寫
    data = read_text_file(path_source)
    la = []
    for line in data:
        la.append(line.upper())
    write_liste_in_file(path_destination, la)

def write_line_in_file(line, path='data/latin_comments.csv', with_anti_slash=True):
    # 將行寫入檔案
    f = open(path, "a+", encoding='utf-8')

    if with_anti_slash:
        f.write(str(line) + "\n")
    else:
        f.write(line)

def list_to_string(liste):
    liste_b = []
    for p in liste:
        if type(p) is not str:
            liste_b.append(str(p))
        else:
            liste_b.append(p)
    return "".join(liste_b)

def check_all_elements_type(list_to_check, types_tuple):
    # 檢查列表中所有元素的類型是否與指定的元組相符
    return all(isinstance(p, types_tuple) for p in list_to_check)

def list_all_files_in_folder(folder_path):
    # 列出資料夾中的所有檔案
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

def get_mnist_as_dataframe():
    """image_list = ch.get_reshaped_matrix(np.array([ch.get_reshaped_matrix(p, (1, 28 * 28)) for p in x_train]),
                                        (x_train.shape[0], 28 * 28))"""

def is_empty_line(string_in):
    string_in = str(string_in)
    if re.match(r'^\s*$', string_in):
        return True
    return False

def write_row_csv(row_liste, file_name='data/latin_comments.csv', delimiter=',', quotechar='`'):
    # 將行寫入 CSV 檔案
    file = open(file_name
