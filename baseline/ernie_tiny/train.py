import paddle
import paddlehub as hub
from pretrain import MyDataset

model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=len(MyDataset.label_list))
tokenizer = model.get_tokenizer()

# 实例化训练集
train_dataset = MyDataset(tokenizer)
dev_dataset = MyDataset(tokenizer=tokenizer, max_seq_len=128, mode='dev')
test_dataset = MyDataset(tokenizer=tokenizer, max_seq_len=128, mode='test')
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls')
# dev dataset

trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, log_interval=5)
trainer.evaluate(test_dataset, batch_size=32)
