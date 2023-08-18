from scheduler_LM import load_scheduler
from load_dataset_LM import load_data
from hyperparams_LM import set_args
from utils_LM import*

def bos_creation(tokens):
    bos_tensor = tf.constant(args.bos_idx, dtype=tokens.dtype)
    batch_size = tf.shape(tokens)[0]
    bos_tokens = tf.tile(tf.reshape(bos_tensor, [1, 1]), [batch_size, 1])

    return bos_tokens


def bos_addition(tokens):
    bos_tensor = tf.constant(args.bos_idx, dtype=tokens.dtype)
    batch_size = tf.shape(tokens)[0]
    bos_tokens = tf.tile(tf.reshape(bos_tensor, [1, 1]), [batch_size, 1])
    new_tokens = tf.concat([bos_tokens, tokens], axis=1)
    new_tokens = new_tokens[:,:-1]  # 暫定処理！batch lengthを可変長に変更した際に要注意！

    return new_tokens


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, pad_idx, **kwargs):
        super(Accuracy, self).__init__(name='seq_acc', **kwargs)
        self.correct_sequences = self.add_weight(name="correct_sequences", initializer="zeros")
        self.total_sequences = self.add_weight(name="total_sequences", initializer="zeros")
        self.pad_idx = pad_idx

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to labels and ensure they match the dtype of y_true
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_labels = tf.cast(y_pred_labels, dtype=y_true.dtype)

        # Check where predictions match the true labels and where y_true is not padding
        correct_preds = tf.equal(y_true, y_pred_labels)
        valid_entries = tf.not_equal(y_true, self.pad_idx)

        # Only consider sequences where all predictions are correct and not padding
        all_correct_seqs = tf.reduce_all(tf.logical_or(correct_preds, tf.logical_not(valid_entries)), axis=-1)

        # Update counts
        self.correct_sequences.assign_add(tf.reduce_sum(tf.cast(all_correct_seqs, tf.float32)))
        self.total_sequences.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.correct_sequences / self.total_sequences

    def reset_state(self):
        self.correct_sequences.assign(0.)
        self.total_sequences.assign(0.)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, if_train):
        super().__init__()
        self.if_train = if_train
        self.layer = tf.keras.layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.k_dim, dropout=args.dropout)
        self.fefw1 = tf.keras.layers.Dense(args.k_dim*2, use_bias=False, activation='swish')
        self.fefw2 = tf.keras.layers.Dense(args.k_dim, use_bias=False)
        self.norma = tf.keras.layers.BatchNormalization(momentum=0.95)
        self.Resid = tf.keras.layers.Add()

    def call(self, q, v, mask, causal=False):
        output_m = self.layer(q, v, attention_mask=mask, training=self.if_train, use_causal_mask=causal)
        output_m = self.Resid([output_m, q])
        output_m = self.norma(output_m, self.if_train)

        if not causal:
            output_f = self.fefw1(output_m)
            output_f = self.fefw2(output_f)
            output_f = self.Resid([output_f, output_m])
            output_f = self.norma(output_f, self.if_train)
        else:
            output_f = output_m

        return output_f


def encoder(x, sou_mask, if_train):
    x = TransformerBlock(if_train)(x, x, sou_mask)
    x = TransformerBlock(if_train)(x, x, sou_mask)

    return x


def decoder(l, x, tar_mask, van_mask, if_train):
    l = TransformerBlock(if_train)(l, l, tar_mask, True)
    l = TransformerBlock(if_train)(l, x, van_mask)
    l = TransformerBlock(if_train)(l, l, tar_mask, True)
    l = TransformerBlock(if_train)(l, x, van_mask)

    return l


def make_mask(q_seq, k_seq):
    q_mask = tf.where(q_seq==args.pad_idx, 0, 1)
    k_mask = tf.where(k_seq==args.pad_idx, 0, 1)
    q_mask = tf.expand_dims(q_mask, -1)
    k_mask = tf.expand_dims(k_mask, -1)
    k_mask = tf.transpose(k_mask, [0, 2, 1])
    atten_mask = tf.matmul(q_mask, k_mask)
    atten_mask = tf.cast(tf.where(atten_mask==0, 0, 1), dtype=tf.int32)

    return atten_mask


def make_embe(x_tokens, l_tokens):
    embed_layer = tf.keras.layers.Embedding(args.vocab_size, args.k_dim)
    posit_layer = tf.keras.layers.Embedding(args.max_st_len, args.k_dim)  # N Len C

    x_embed = embed_layer(x_tokens)  # N Len C
    p_index = tf.tile(tf.expand_dims(tf.range(tf.shape(x_tokens)[-1]), 0), [tf.shape(x_tokens)[0], 1])
    x_posit = posit_layer(p_index)

    l_embed = embed_layer(l_tokens)  # N Len C
    p_index = tf.tile(tf.expand_dims(tf.range(tf.shape(l_tokens)[-1]), 0), [tf.shape(l_tokens)[0], 1])
    l_posit = posit_layer(p_index)

    x = x_embed + x_posit
    l = l_embed + l_posit

    return x, l, embed_layer


def crossentropy(labels, logits):
    idxs = tf.where(labels != args.pad_idx)
    labels = tf.gather_nd(labels, idxs)
    logits = tf.gather_nd(logits, idxs)

    # One Hot Encode Sparsely Encoded Target Sign
    labels = tf.one_hot(labels, args.vocab_size, axis=1)

    # Categorical Crossentropy with native label smoothing support
    loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True, label_smoothing=args.smoothing)
    loss = tf.math.reduce_mean(loss)

    return loss


def get_model(if_train=True):
    INPUT_SHAPE = (None,)
    input = tf.keras.Input(INPUT_SHAPE)
    x     = input  # N Len

    if if_train:
        label = tf.keras.Input(INPUT_SHAPE)
        l     = label            # N Len
        l     = bos_addition(l)  # shifted right
    else:
        l     = bos_creation(x)  # N 1

    sou_mask = make_mask(x, x)   # for pred seq 
    tar_mask = make_mask(l, l)   # for label seq
    van_mask = make_mask(l, x)   # for vanilla
    x, l, sw = make_embe(x, l)   # N Len C

    x = encoder(x, sou_mask, if_train)
    l = decoder(l, x, tar_mask, van_mask, if_train)
    l = tf.keras.layers.Dropout(0.1)(l, training=if_train)
    logits = tf.einsum('nlc,cv->nlv', l, tf.transpose(sw.weights[0]))  # N, Len, vocab_size

    if if_train:
        model = tf.keras.Model((input, label), logits)
    else:
        model = tf.keras.Model(input, logits)

    return model


def load_model():
    # Clear all models in GP
    gc.collect()
    tf.keras.backend.clear_session()
    model = get_model()
    optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
    optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)
    model.compile(loss=crossentropy, metrics=Accuracy(pad_idx=args.pad_idx), optimizer=optimizer)
    model.summary()

    return model


def train(model, pack):
    # Actual Training
    history = model.fit(
            x               =pack.get("x"),
            epochs          =pack.get("epochs"),
            validation_data =pack.get("validation_data"),
            validation_steps=pack.get("validation_steps"),
            callbacks       =pack.get("callbacks"),
            verbose         =pack.get("verbose", 'auto'),)

    # Save model weights
    model.save_weights('{}/model.h5'.format(args.save_path))

    # Save traing plot
    save_plot(history, args.save_path)


def train_LM():
    device_assign(args)

    # Load data
    pack = load_data(args)

    # Get new fresh model
    model = load_model()

    # Setting train scheduler
    load_scheduler(args, pack, model)

    # Start training
    train(model, pack)


if __name__ == '__main__':
    args = set_args()
    train_LM()