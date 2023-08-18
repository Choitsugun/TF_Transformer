import argparse

def set_args():
    parser = argparse.ArgumentParser()

    #========== environ ==========#
    parser.add_argument('--device', default='1', type=str)

    #========== dataset ==========#
    parser.add_argument('--dataset_root',    default="/notebooks/dataset", type=str)
    

    #=========== Model ===========#
    parser.add_argument('--k_dim',      default=256, type=int)
    parser.add_argument('--num_heads',  default=4,   type=int)
    parser.add_argument('--dropout',    default=0.2, type=float)
    parser.add_argument('--pad_idx',    default=59,  type=int)
    parser.add_argument('--bos_idx',    default=60,  type=int)
    parser.add_argument('--eos_idx',    default=61,  type=int)
    parser.add_argument('--vocab_size', default=62,  type=int)
    parser.add_argument('--max_st_len', default=45,  type=int)
    parser.add_argument('--smoothing',  default=0.05, type=float)

    #========= scheduler =========#
    parser.add_argument('--CHECKPOINT',    default="/notebooks/save/LM/model.h5", type=str)
    parser.add_argument('--save_path',     default="/notebooks/save/LM",          type=str)
    parser.add_argument('--USE_VAL',       default=True,       type=bool)
    parser.add_argument('--patience',      default=10,         type=int)
    parser.add_argument('--monitor',       default='val_loss', type=str)
    parser.add_argument('--batch_size',    default=128,        type=int)
    parser.add_argument('--N_EPOCHS',      default=200,        type=int)
    parser.add_argument('--N_WARMUP',      default=2,          type=int)
    parser.add_argument('--LR_MAX',        default=0.5e-5,     type=float)
    parser.add_argument('--WD_RATIO',      default=0.05,       type=float)
    parser.add_argument('--WARMUP_METHOD', default="exp",      type=str)
    parser.add_argument('--verbose',       default=1,          type=int)

    return parser.parse_args()