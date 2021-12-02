class AverageMeter(object):
    """
        Computes and stores the average and current value
        入参是记录的名字和想要输出的格式（浮点，指数之类的）
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """
        入参：
            num_batches：batch的多少，就是总的数据//batch_size
            meters：就是要打印的所有参数，比如loss，acc，放在一个列表里
            prefix：就是一个字符串，要放在打印的最前面的（如epoch）
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
            打印的信息是：
                打印在最前面的（如epoch）+ batch信息 + 要打印的所有参数
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
            获取打印的batch的字符串格式而已
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

batch_size = 8
loss_all = [[1.5, 1.34, 1.10, 0.963, 0.89, 0.61], [1.5, 1.34, 1.10, 0.963, 0.89, 0.61]]
top_all = [[6.5, 7.5, 8.5, 9.5, 9.6, 9.9], [6.5, 7.5, 8.5, 9.5, 9.6, 9.9]]

for epoch in range(2):
    losses = AverageMeter('Loss', ':.4e')       # 保留小数点后4位，用指数的形式
    top1 = AverageMeter('Acc@1', ':6.2f')       # 保留小数点后2位，用浮点数形式。6是指定输出宽度。假设输出位数大于宽度，就按照位数输出，否则用空格补全到宽度
    progress = ProgressMeter(6, [losses, top1], prefix="Epoch: [{}]".format(epoch + 1))

    for batch in range(6):
        losses.update(loss_all[epoch][batch], batch_size)
        top1.update(top_all[epoch][batch], batch_size)
        progress.display(batch + 1)
