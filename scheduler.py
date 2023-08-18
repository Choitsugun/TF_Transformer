from Levenshtein import distance
from utils_LM import*

def num_to_char_fn_rev(y):
    y= y[y != 59]

    return [num_to_char.get(x, "") for x in y]

def num_to_char_fn(y):
    indices = np.where(y==61)

    if indices[0].size > 0:
        first_index = indices[0][0]
        y = y[:first_index]

    return [num_to_char.get(x, "") for x in y]


@tf.function()
def decode_phrase(pred):
    x = tf.argmax(pred, axis=-1)
    return x


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, wd_ratio):
        self.step_counter = 0
        self.model = model
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')


# A callback class to output a few transcriptions during training
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    def __init__(self, model, dataset):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, epoch: int, logs=None):
        if self.dataset is None:
            return

        predictions = []
        targets = []
        tar_rev = []
        dis_sum = 0
        for batch in self.dataset:
            X, y = batch
            a, _ = X
            batch_predictions = self.model(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = "".join(num_to_char_fn(label.numpy()))
                targets.append(label)
            for rev in a:
                rev = "".join(num_to_char_fn_rev(rev.numpy()))
                tar_rev.append(rev)
        print("-" * 100)
        # for i in np.random.randint(0, len(predictions), 2):
        for i in range(batch_size):
            print(f"Target    : {targets[i]}")
            print(f"Tar_re    : {tar_rev[i]}")
            print(f"Prediction: {predictions[i]}, len: {len(predictions[i])}")
            print("-" * 100)

        for i in range(batch_size):
            inp_str = predictions[i]
            label_str = "".join(num_to_char_fn(y[i].numpy()))
            print("Predi:", inp_str)
            print("label:", label_str)
            dis = distance(inp_str, label_str)
            print("distance:", dis)
            dis_sum = dis_sum + dis

        print("mean distance:", dis_sum/batch_size)


def lrfn(current_step, num_warmup_steps, lr_max, num_cycles):
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, N_EPOCHS - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max


def load_scheduler(args, pack, model):
    global num_to_char, pad_token_idx, batch_size, save_root, N_EPOCHS,\
           WARMUP_METHOD, N_WARMUP_EPOCHS, LR_MAX, WD_RATIO

    num_to_char = args.num_to_char
    pad_token_idx = args.pad_idx
    batch_size = args.batch_size
    save_root = args.save_path
    N_EPOCHS = args.N_EPOCHS
    WARMUP_METHOD = args.WARMUP_METHOD
    N_WARMUP_EPOCHS = args.N_WARMUP
    LR_MAX = args.LR_MAX
    WD_RATIO = args.WD_RATIO

    # Make Callback Function
    pack["epochs"] = N_EPOCHS
    pack["verbose"] = args.verbose

    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.5) for step in range(N_EPOCHS)]

    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

    # Early stopping Callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=args.monitor, patience=args.patience, restore_best_weights=True)

    # Callback function to check transcription on the val set.
    validation_callback = CallbackEval(model, pack["validation_data"].take(1) if args.USE_VAL else None)

    pack["callbacks"] = [lr_callback, WeightDecayCallback(model, WD_RATIO), early_stop, validation_callback]