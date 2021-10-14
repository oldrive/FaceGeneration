import datetime
from time import sleep

import numpy as np
import wandb

from options.train_options import TrainOptions
from utils import common


class WBLogger:

    def __init__(self, opts):
        wandb_run_name = opts.dataset_type
        wandb.init(project="id_disentangle", config=vars(opts), name=wandb_run_name)

    @staticmethod
    def log_best_model():
        wandb.run.summary["best-model-save-time"] = datetime.datetime.now()

    @staticmethod
    def log(prefix, metrics_dict, global_step):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
        wandb.log(log_dict)
        # wandb.log(log_dict, step=global_step)

    @staticmethod
    def log_dataset_wandb(dataset, dataset_name, n_images=16):
        idxs = np.random.choice(a=range(len(dataset)), size=n_images, replace=False)
        data = [wandb.Image(dataset.source_paths[idx]) for idx in idxs]
        wandb.log({f"{dataset_name} Data Samples": data})

    # @staticmethod
    # def log_images_to_wandb(x, y, y_hat, id_logs, prefix, step, opts):
    #     im_data = []
    #     column_names = ["Source", "Target", "Output"]
    #     if id_logs is not None:
    #         column_names.append("ID Diff Output to Target")
    #     for i in range(len(x)):
    #         cur_im_data = [
    #             wandb.Image(common.log_input_image(x[i], opts)),
    #             wandb.Image(common.tensor2im(y[i])),
    #             wandb.Image(common.tensor2im(y_hat[i])),
    #         ]
    #         if id_logs is not None:
    #             cur_im_data.append(id_logs[i]["diff_target"])
    #         im_data.append(cur_im_data)
    #     outputs_table = wandb.Table(data=im_data, columns=column_names)
    #     wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})

    @staticmethod
    def log_images_to_wandb(prefix, images, step, generated=False, postfixs=None):
        '''

        :param prefix:
        :param images: 值在[-1.0, 1.0]之间
        :param step:
        :param generated:传进来的图片是生成的还是数据集中的
        :return:
        '''
        im_data = []
        size = images.shape[0]
        if generated:
            im_names = [prefix + postfix for postfix in postfixs]
        else:
            im_names = [prefix + str(i) for i in range(images.shape[0])]
        for i in range(size):
            im_data.append(wandb.Image(common.tensor2im(images[i]), caption=im_names[i]))
        wandb.log({f"{prefix.title()}": im_data})


if __name__ == '__main__':
    opts = TrainOptions().parse()
    w = WBLogger(opts)
    dict = {}
    dict['epoch'] = 1
    w.log('train', dict, 0)

    print('have done')

    for i in range(1, 101):
        print(i)
        sleep(i)




