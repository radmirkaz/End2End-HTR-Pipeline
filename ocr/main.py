import os
import shutil

from config import config
from utils import set_seed, run, inference


if __name__ == '__main__':
    set_seed(seed=config.general.seed)

    if not config.inference.inference:
        if config.data.kfold.train_all_folds:
            config.training.verbose_plots = False

            for x in range(config.data.kfold.params.n_splits):
                config.data.kfold.current_fold = x
                run(config)

                os.rename('submission/checkpoints/best.pt', f'submission/checkpoints/fold_{x + 1}.pt')
                shutil.move(f'submission/checkpoints/fold_{x + 1}.pt', f'submission/fold_{x + 1}.pt')

                os.rename('submission/checkpoints/log.txt', f'submission/checkpoints/log_{x + 1}.txt')
                shutil.move(f'submission/checkpoints/log_{x + 1}.txt', f'log_{x + 1}.txt')

                shutil.rmtree('submission/checkpoints/')
        else:
            run(config)
    else:
        inference(config)

