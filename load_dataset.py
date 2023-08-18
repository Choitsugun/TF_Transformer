import pandas as pd
import tensorflow as tf
import re
import random 
import json
from hyperparams_LM import set_args

def is_phone(phrase):
    # Phone Number
    if re.match(r'^[\d+-]+$', phrase):
        return True
    else:
        return False

def create_data_gen(phrase_list, max_phrase_len, pad_idx, eos_idx, char_to_num):
    def gen():
        for true_phrase in phrase_list:
            # fake_phraseの作成
            fake_phrase = random_delete(true_phrase)
            # tokenize
            inp   = [char_to_num[x] for x in fake_phrase] + [eos_idx]
            #inp  = [char_to_num[x] for x in true_phrase] + [eos_idx]
            inp_q = [char_to_num[x] for x in true_phrase]
            label = [char_to_num[x] for x in true_phrase] + [eos_idx]
            # padding
            inp   = inp   + [pad_idx]*(max_phrase_len - len(inp))
            inp_q = inp_q + [pad_idx]*(max_phrase_len - len(inp_q))
            label = label + [pad_idx]*(max_phrase_len - len(label))

            yield (inp, inp_q), label
    return gen

def random_delete(s):
    # 文字列が空の場合はそのまま返す
    if not s:
        return s

    # ランダムな位置を選択
    index = random.randint(0, len(s) - 1)

    # 選ばれた位置の文字を消去
    if s[index].isdigit():
        return s
    else:
        return s[:index] + s[index+1:]

def get_dataset(phrase_list, max_phrase_len, pad_idx, eos_idx, char_to_num, batch_size):
    ds =tf.data.Dataset.from_generator(
        create_data_gen(
            phrase_list, max_phrase_len, pad_idx, eos_idx, char_to_num),
        output_signature=((tf.TensorSpec(shape=(max_phrase_len), dtype=tf.int8),
                           tf.TensorSpec(shape=(max_phrase_len), dtype=tf.int8)),
                           tf.TensorSpec(shape=(max_phrase_len), dtype=tf.int8)))

    ds = ds.prefetch(buffer_size=2000).batch(batch_size)
    return ds


def load_data(args):
    with open (f"{args.dataset_root}/character_to_prediction_index.json", "r") as f:
        char_to_num = json.load(f)

    char_to_num["pad"] = args.pad_idx
    char_to_num["bos"] = args.bos_idx
    char_to_num["eos"] = args.eos_idx
    assert args.vocab_size == len(char_to_num)
    num_to_char = {j:i for i,j in char_to_num.items()}
    args.num_to_char = num_to_char

    max_phrase_len = args.max_st_len
    pad_idx = args.pad_idx
    eos_idx = args.eos_idx

    df = pd.read_csv(f"{args.dataset_root}/train.csv")
    df["is_phone"] = df['phrase'].apply(is_phone)
    df = df[df["is_phone"] == False]
    phrase_list = df["phrase"].tolist()
    val_len = int(len(phrase_list)*0.05)

    train_dataset = get_dataset(phrase_list[val_len:], max_phrase_len, pad_idx, eos_idx, char_to_num, args.batch_size)
    val_dataset = get_dataset(phrase_list[:val_len], max_phrase_len, pad_idx, eos_idx, char_to_num, args.batch_size)
    pack = {"x":train_dataset, "validation_data": val_dataset if args.USE_VAL else None}

    return pack

def num_to_char_fn(y, num_to_char):
    return "".join([num_to_char.get(x, "") for x in y])

if __name__ == "__main__":
    args = set_args()
    pack = load_data(args)
    _, val_dataset = pack["x"], pack["validation_data"]

    with open (f"{args.dataset_root}/character_to_prediction_index.json", "r") as f:
        char_to_num = json.load(f)
    char_to_num["pad"] = args.pad_idx
    char_to_num["bos"] = args.bos_idx
    char_to_num["eos"] = args.eos_idx

    num_to_char = {j:i for i,j in char_to_num.items()}
    for input, label in val_dataset:
        inp, inp_q = input
        inp = inp.numpy()
        inp_q = inp_q.numpy()
        label = label.numpy()
        for i in range(args.batch_size):
            print("inp__:", num_to_char_fn(inp[i], num_to_char))
            print("inp_q:", num_to_char_fn(inp_q[i], num_to_char))
            print("label:", num_to_char_fn(label[i], num_to_char))
            print()
        break




