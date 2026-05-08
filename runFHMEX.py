import argparse
from src.train.trainer import load_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='FreqMOE_V3', type=str)
    parser.add_argument('--model_type', default='General', choices=['General', 'LLM-based'], type=str)
    parser.add_argument('--num_expert', default=2, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--kernel_size', default=11, type=int)
    parser.add_argument('--num_bands', default=6, type=int)
    parser.add_argument('--freq_dropout_prob', default=0.4, type=float)
    parser.add_argument('--lamda1', default=0.1, type=float)
    parser.add_argument('--lamda2', default=0.1, type=float)

    # Data
    parser.add_argument('--dataset', default='weibo21', type=str)
    # Image data
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    # Text data
    parser.add_argument('--max_text_len', default=197, type=int, help='max sentence length')
    # Training
    parser.add_argument('--epoch_num', default=1, type=int)
    parser.add_argument('--train_batch', default=48, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_worker', default=4, type=int, help='num_workers for dataloader')
    parser.add_argument('--l2', default=0.01, type=float, help='l2 normalization')
    parser.add_argument('--patience', default=50, type=int, help='early stop patience')
    parser.add_argument('--seed', default=2027, type=int)
    parser.add_argument('--device', default='cuda:0', help='training on gpu or cpu, default gpu')
    parser.add_argument('--test_device', default='cpu', type=str, help='do test on cpu or cuda')
    parser.add_argument('--mark', default='save-news-emb', type=str)
    # Evaluation
    parser.add_argument('--eval_batch', default=24, type=int)
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='PS', type=str, help='[LS, LS_R@0.x, PS]')
    parser.add_argument('--metric', default=['acc', 'precision', 'recall', 'f1'], type=str, nargs='+')
    parser.add_argument('--valid_metric', default='acc', help='specifies which indicator to apply early stop')

    config = parser.parse_args()
    trainer = load_trainer(config)
    trainer.start_training()
