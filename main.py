from executer import Execute
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)
def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='dataset')
    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage')
    parser.add_argument("--eval_dataset", type=str, default='BPDP',
                        help="Available datasets: FactBench, BPDP")
    parser.add_argument("--subpath", type=str, default='bpdp/')
    parser.add_argument("--prop", type=str, default=None)
    parser.add_argument("--cmp_dataset", type=bool, default=False)
    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')


    # Models.
    parser.add_argument("--model", type=str, default='full-Hybrid',
                        help="Available models:full-Hybrid, KGE-only, text-KGE-Hybrid, path-only, text-path-Hybrid, KGE-path-Hybrid")
                        # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    parser.add_argument("--emb_type", type=str, default='TransE',
                        help="Available KG embeddings: ConEx, TransE, ComplEx, RDF2Vec, QMult")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=1000)
    parser.add_argument("--min_num_epochs", type=int, default=200)
    # parser.add_argument('--batch_size', type=int, default=345)
    # parser.add_argument('--val_batch_size', type=int, default=345)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)



if __name__ == '__main__':
    args = argparse_default()
    if args.eval_dataset == "FactBench":
        datasets_class = [ "range/","property/", "domain/", "domainrange/", "mix/", "random/"]
        for cls in datasets_class:
            args = argparse_default()
            args.subpath = cls
            exc = Execute(args)
            exc.start()
            # exit(1)
    elif args.eval_dataset=="BPDP":
        exc = Execute(args)
        exc.start()
        exit(1)
    else:
        print("Please specify the dataset")
        exit(1)




