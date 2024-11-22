import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, f1_score
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer,MixedTemplate
from openprompt import PromptForClassification
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from pacing_functions import PACING_FUNCTIONS  # 引入 pacing_functions

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pacing_function="root_10"
seed = 42
batch_size = 16
num_class = 4
max_seq_l = 512
lr = 5e-5
num_epochs = 100
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "E:\\models\\unixcoder-base"
early_stop_threshold = 10

classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def read_data_to_dataframe(filename,is_train=False):
    data = pd.read_excel(filename).astype(str)
    if is_train:
        # 对训练集按照 'score' 列进行排序
        data = data.sort_values(by='score')
    return data[['abstract_func_before', 'description', 'severity']]

def convert_dataframe_to_dataset(data):
    examples = {
        'text_a': [],
        'text_b': [],
        'label': []
    }
    for idx, row in data.iterrows():
        examples['text_a'].append(' '.join(row['abstract_func_before'].split(' ')[:384]))
        examples['text_b'].append(' '.join(row['description'].split(' ')[:64]))
        examples['label'].append(int(row['severity']))
    return Dataset.from_dict(examples)

# Read and convert data
train_data = read_data_to_dataframe(r"E:\wjy\ordinal regression\code\complete_test_data_with_absolute_errors.xlsx", is_train=True)
valid_data = read_data_to_dataframe(r"C:\Users\Admin\Desktop\data_c++\\valid.xlsx",is_train=False)
test_data = read_data_to_dataframe(r"C:\Users\Admin\Desktop\data_c++\\test.xlsx",is_train=False)

train_dataset = convert_dataframe_to_dataset(train_data)
valid_dataset = convert_dataframe_to_dataset(valid_data)
test_dataset = convert_dataframe_to_dataset(test_data)

# Create the splits dictionary
train_val_test = {
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
}

# Convert to InputExample format
dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in train_val_test[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset[split].append(input_example)

# Load PLM
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", pretrainedmodel_path)

# Construct template
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description: {"placeholder":"text_b"} {"soft":"Classify the severity:"} {"mask"}'
mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text,model=plm)
# 定义损失函数
class CORALOrdinalLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(CORALOrdinalLoss, self).__init__()
        self.num_classes = num_classes - 1

    def forward(self, logits, labels):
        labels_bin = torch.zeros(labels.size(0), self.num_classes).to(logits.device)
        for i in range(self.num_classes):
            labels_bin[:, i] = (labels > i).float()
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss = loss_func(logits, labels_bin)
        return loss


# 定义模型
class CORALPromptModelWithLinear(nn.Module):
    def __init__(self, prompt_model, num_classes):
        super(CORALPromptModelWithLinear, self).__init__()
        self.prompt_model = prompt_model
        self.num_classes = num_classes - 1
        self.linear = nn.Linear(num_classes, self.num_classes)

    def forward(self, inputs):
        logits = self.prompt_model(inputs)
        logits = self.linear(logits)
        return logits


# 测试模型
def test(coral_model, test_dataloader, name):
    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = coral_model(inputs)
            labels = inputs['label']
            progress_bar.update(1)
            preds = (logits > 0).sum(dim=-1).cpu().tolist()
            allpreds.extend(preds)
            alllabels.extend(labels.cpu().tolist())

        acc = accuracy_score(alllabels, allpreds)
        spearman_corr, _ = spearmanr(alllabels, allpreds)
        mse = sum((p - l) ** 2 for p, l in zip(allpreds, alllabels)) / len(alllabels)
        mze = sum(p != l for p, l in zip(allpreds, alllabels)) / len(alllabels)
        macro_f1 = f1_score(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)

        with open(os.path.join('./results', "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join('./results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')

        print(
            f"acc: {acc}   Spearman Correlation: {spearman_corr}   MSE: {mse}   MZE: {mze}   Macro F1: {macro_f1}   MCC: {mcc}")

    return acc, spearman_corr, mse, mze, macro_f1, mcc
# DataLoaders
train_dataloader = PromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                    predict_eos_token=False, truncate_method="head", decoder_max_length=3)
validation_dataloader = PromptDataLoader(dataset=dataset['validation'], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                         predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader = PromptDataLoader(dataset=dataset['test'], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                   batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                   predict_eos_token=False, truncate_method="head", decoder_max_length=3)

# Verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={"LOW": ["low", "slight"],
                                             "MEDIUM": ["medium", "moderate"],
                                             "HIGH": ["high", "severe"],
                                             "CRITICAL": ["critical", "significant"]})
max_steps = num_epochs * len(train_dataloader)/2
save_steps = len(train_dataloader)
warmup_steps = len(train_dataloader)
num_train_epochs = num_epochs
# Prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# Optimizer parameters
no_decay = ['bias', 'LayerNorm.weight']
# 初始化模型
coral_model = CORALPromptModelWithLinear(
    PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False), num_class)
if use_cuda:
    coral_model = coral_model.cuda()
coral_loss_func = CORALOrdinalLoss(num_class)
# 设置优化器和学习率调度器
optimizer1 = AdamW(coral_model.parameters(), lr=lr)
optimizer2 = AdamW(
        [{'params': [p for n, p in coral_model.prompt_model.template.named_parameters() if "raw_embedding" not in n],'weight_decay': 0.01}],
        lr=5e-5)
scheduler1 = get_linear_schedule_with_warmup(optimizer1,
                                                 num_training_steps=num_epochs * len(train_dataloader),num_warmup_steps=num_epochs * len(train_dataloader))
scheduler2 = get_linear_schedule_with_warmup(optimizer2,
                                                 num_training_steps=num_epochs * len(train_dataloader),
                                                 num_warmup_steps=num_epochs * len(train_dataloader))
# optimizer_grouped_parameters1 = [
# Test function


# Training and evaluation
output_dir = "vultypeprompt3_log"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
global_step = 0
all_step = 0
tr_loss = 0.0
best_metric = 1e9
c0 = 0.33
now_percent = 0
early_stop_count = 0
skip_early_stop_count = 0
early_stop_threshold = 10
num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))
# 早停参数
patience = 10  # 早停耐心值
early_stop_count = 0  # 早停计数器
best_metric = float('inf')  # 初始化最佳指标为正无穷大
subset_percentage = 0.33  # 初始子集比例，可根据需求调整

for epoch in range(num_epochs):
    early_stop_count = 0  # 早停计数器
    # 计算当前子集数据量
    subset_length = int(len(dataset["train"]) * subset_percentage)
    train_data_subset = dataset["train"][:subset_length]

    # 创建训练数据加载器
    train_dataloader = PromptDataLoader(
        dataset=train_data_subset,
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=512,
        batch_size=batch_size,
        shuffle=True,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
        decoder_max_length=3
    )

    for inner_epoch in range(num_epochs):  # 每个子集的训练循环
        coral_model.train()
        tot_loss = 0

        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = coral_model(inputs)
            labels = inputs['label'].cuda()
            loss = coral_loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            scheduler1.step()
            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()
            progress_bar.update(1)
            all_step += 1
            if (step + 1) % 100 == 0:
                avg_loss = tot_loss / (step + 1)
                print(f"Subset {subset_percentage * 100:.1f}%, Epoch: {epoch}, Step: {step + 1}, Loss: {avg_loss}")

        # 验证
        coral_model.eval()
        acc, spearman_corr, mse, mze, macro_f1, mcc = test(coral_model, validation_dataloader,
                                                           f"validation_subset_{subset_percentage * 100:.1f}_epoch_{epoch}")

        # 检查是否有更好的模型
        if mse < best_metric:
            best_metric = mse
            torch.save(coral_model.state_dict(), './best_model.ckpt')
            print(f"Best model saved with mse: {best_metric}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1
            print(f"No improvement in MSE. Early stop counter: {early_stop_count}/{patience}")

        # 检查是否达到早停条件
        if early_stop_count >= patience:
            print("Early stopping for this subset triggered.")
            break  # 停止当前子集的训练

    # 更新到更大的数据子集
    if pacing_function != "":
        curriculum_iterations = num_training_steps/2
        new_data_fraction = min(1, PACING_FUNCTIONS[pacing_function](all_step, curriculum_iterations, c0))
        percent = int(new_data_fraction * len(dataset["train"]))
        if percent != subset_length:
            now_percent = percent
            print(f"Updating training data to {now_percent} samples.")
            early_stop_count=0
    subset_percentage = new_data_fraction  # 增加子集比例


# 检查是否完成全部数据集的训练
    if subset_percentage >= 1.0 and early_stop_count==10:
        print("Training on all data subsets complete.")
        break  # 结束整个训练过程


# 加载最佳模型并测试
coral_model.load_state_dict(torch.load('./best_model.ckpt'))
test(coral_model, test_dataloader, "test")