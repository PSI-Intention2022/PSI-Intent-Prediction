import numpy as np


def save_args(self, args):
    # 3. args
    with open(args.checkpoint_path + '/args.txt', 'w') as f:
        for arg in vars(self.args):
            val = getattr(self.args, arg)
            if isinstance(val, str):
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    np.save(self.args.checkpoint_path + "/args.npy", self.args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

