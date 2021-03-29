from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset


class MyDataset(TextClassificationDataset):
    # 数据集存放目录
    base_path = 'data/'
    # 数据集的标签列表
    label_list = ['negative', 'positive']

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'dev.txt'
        super().__init__(
            base_path=self.base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=self.label_list,
            is_file_with_header=True)
