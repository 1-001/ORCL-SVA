import os
import torch
import pandas as pd
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from config import num_class
from pacing_functions import PACING_FUNCTIONS  # 引入 pacing_functions

num_class=4
num_epochs = 100
learning_rate = 5e-5
lr = 5e-5

# 定义参数类
class Args:
    def __init__(self):
        self.pacing_function = 'root_10'  # 可以选择 pacing 函数
        self.batch_size = 16
        self.max_seq_length = 512
        self.use_cuda = True
        self.seed = 42

# Define classes
classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
args = Args()

# 设置设备
if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model_name = "codet5"
pretrainedmodel_path = "E:\\models\\codet5-base"

# 读取数据
def read_prompt_examples(filename, is_train=False):
    examples = []

    data = pd.read_excel(filename).astype(str)
    # 将 DataFrame 转换为字典列表，然后根据 'score' 排序

    if is_train:
        # 对训练集按照 'score' 列进行排序
        data = data.sort_values(by='score')
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    severity = data['severity'].tolist()
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(severity[idx]),
            )
        )
    return examples


# 加载模型和模板
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description: {"placeholder":"text_b"} Classify the severity: {"mask"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

# 定义 Verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    "LOW": ["low", "slight"],
                                    "MEDIUM": ["medium", "moderate"],
                                    "HIGH": ["high", "severe"],
                                    "CRITICAL": ["critical", "significant"]
                                })


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
            if args.use_cuda:
                inputs = inputs.cuda()
            logits = coral_model(inputs)
            labels = inputs['tgt_text']
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


# 主程序
traination_data = read_prompt_examples(r"E:\wjy\ordinal regression\code\complete_test_data_with_absolute_errors.xlsx", is_train=True)
train_dataloader = PromptDataLoader(
    dataset=traination_data,
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    shuffle=True,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="head",
    decoder_max_length=3
)
max_steps = num_epochs * len(train_dataloader)/2
save_steps = len(train_dataloader)
warmup_steps = len(train_dataloader)
num_train_epochs = num_epochs
# 加载验证集
validation_data = read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\valid.xlsx", is_train=False)
validation_dataloader = PromptDataLoader(
    dataset=validation_data,
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    shuffle=True,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="head",
    decoder_max_length=3
)

# 加载测试集
test_data = read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\test.xlsx", is_train=False)
test_dataloader = PromptDataLoader(
    dataset=test_data,
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="head",
    decoder_max_length=3
)
# Optimizer parameters
no_decay = ['bias', 'LayerNorm.weight']
# 初始化模型
coral_model = CORALPromptModelWithLinear(
    PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False), num_class)
if args.use_cuda:
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
#     {'params': [p for n, p in coral_model.parameters() if not any(nd in n for nd in no_decay)],
#      'weight_decay': 0.01},
#     {'params': [p for n, p in coral_model.parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
# ]
# # Using different optimizer for prompt parameters and model parameters
# optimizer_grouped_parameters2 = [
#     {'params': [p for n, p in coral_model.prompt_model.template.named_parameters() if "raw_embedding" not in n]}
# ]
# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)  # learning rate for model parameters
# optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)  # learning rate for prompt parameters
#
# num_training_steps = num_epochs * len(train_dataloader)
# scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0,
#                                              num_training_steps=num_training_steps)  # set warmup steps
# scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0,
#                                              num_training_steps=num_training_steps)  # set warmup steps
# 训练
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
# # 读取和排序数据
# data = pd.read_excel(r"E:\wjy\ordinal regression\code\complete_test_data_with_absolute_errors.xlsx").sort_values(by='score')
#
# # 计算前 33% 的样本数
# num_samples = int(len(data) * 0.33)
#
# # 提取前 33% 的数据并将其转为 InputExample 格式
# subset_data = data.iloc[:num_samples]
# examples = []
# desc = subset_data['description'].tolist()
# code = subset_data['abstract_func_before'].tolist()
# severity = subset_data['severity'].tolist()
# for idx in range(len(subset_data)):
#     examples.append(
#         InputExample(
#             guid=idx,
#             text_a=' '.join(code[idx].split(' ')[:384]),
#             text_b=' '.join(desc[idx].split(' ')[:64]),
#             tgt_text=int(severity[idx]),
#         )
#     )
#
# # 创建新的训练数据加载器
# train_dataloader = PromptDataLoader(
#     dataset=examples,
#     template=mytemplate,
#     tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass,
#     max_seq_length=args.max_seq_length,
#     batch_size=args.batch_size,
#     shuffle=True,
#     teacher_forcing=False,
#     predict_eos_token=False,
#     truncate_method="tail",
#     decoder_max_length=3
# )

# 早停参数
patience = 10  # 早停耐心值
early_stop_count = 0  # 早停计数器
best_metric = float('inf')  # 初始化最佳指标为正无穷大
subset_percentage = 0.33  # 初始子集比例，可根据需求调整

for epoch in range(num_epochs):
    early_stop_count = 0  # 早停计数器
    # 计算当前子集数据量
    subset_length = int(len(traination_data) * subset_percentage)
    train_data_subset = traination_data[:subset_length]

    # 创建训练数据加载器
    train_dataloader = PromptDataLoader(
        dataset=train_data_subset,
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
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
            if args.use_cuda:
                inputs = inputs.cuda()
            logits = coral_model(inputs)
            labels = inputs['tgt_text'].cuda()
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
    if args.pacing_function != "":
        curriculum_iterations = num_training_steps/2
        new_data_fraction = min(1, PACING_FUNCTIONS[args.pacing_function](all_step, curriculum_iterations, c0))
        percent = int(new_data_fraction * len(traination_data))
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