import os
import torch
import pandas as pd
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer
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


class Args:
    def __init__(self):
        self.pacing_function = 'root_10'  
        self.batch_size = 16
        self.max_seq_length = 512
        self.use_cuda = True
        self.seed = 42

# Define classes
classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
args = Args()


if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model_name = "codet5"
pretrainedmodel_path = "E:\\models\\codet5-base"


def read_prompt_examples(filename, is_train=False):
    examples = []

    data = pd.read_excel(filename).astype(str)
    

    if is_train:
        
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



plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description: {"placeholder":"text_b"} {"soft":"Classify the severity:"} {"mask"}'
mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)


myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    "LOW": ["low"],
                                    "MEDIUM": ["medium"],
                                    "HIGH": ["high"],
                                    "CRITICAL": ["critical"]
                                })



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



traination_data = read_prompt_examples(r"E:\wjy\ordinal regression\code\difficultyscores.xlsx", is_train=True)
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

no_decay = ['bias', 'LayerNorm.weight']

coral_model = CORALPromptModelWithLinear(
    PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False), num_class)
if args.use_cuda:
    coral_model = coral_model.cuda()
coral_loss_func = CORALOrdinalLoss(num_class)

optimizer1 = AdamW(coral_model.parameters(), lr=lr)
optimizer2 = AdamW(
        [{'params': [p for n, p in coral_model.prompt_model.template.named_parameters() if "raw_embedding" not in n],'weight_decay': 0.01}],
        lr=5e-5)
scheduler1 = get_linear_schedule_with_warmup(optimizer1,
                                                 num_training_steps=num_epochs * len(train_dataloader),num_warmup_steps=num_epochs * len(train_dataloader))
scheduler2 = get_linear_schedule_with_warmup(optimizer2,
                                                 num_training_steps=num_epochs * len(train_dataloader),
                                                 num_warmup_steps=num_epochs * len(train_dataloader))

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

patience = 10  
early_stop_count = 0  
best_metric = float('inf')  
subset_percentage = 0.33  

for epoch in range(num_epochs):
    early_stop_count = 0  
    
    subset_length = int(len(traination_data) * subset_percentage)
    train_data_subset = traination_data[:subset_length]

    
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

    for inner_epoch in range(num_epochs):  
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

        
        coral_model.eval()
        acc, spearman_corr, mse, mze, macro_f1, mcc = test(coral_model, validation_dataloader,
                                                           f"validation_subset_{subset_percentage * 100:.1f}_epoch_{epoch}")

        
        if mse < best_metric:
            best_metric = mse
            torch.save(coral_model.state_dict(), './best_model.ckpt')
            print(f"Best model saved with mse: {best_metric}")
            early_stop_count = 0  
        else:
            early_stop_count += 1
            print(f"No improvement in MSE. Early stop counter: {early_stop_count}/{patience}")

        
        if early_stop_count >= patience:
            print("Early stopping for this subset triggered.")
            break  

    
    if args.pacing_function != "":
        curriculum_iterations = num_training_steps/2
        new_data_fraction = min(1, PACING_FUNCTIONS[args.pacing_function](all_step, curriculum_iterations, c0))
        percent = int(new_data_fraction * len(traination_data))
        if percent != subset_length:
            now_percent = percent
            print(f"Updating training data to {now_percent} samples.")
            early_stop_count=0
    subset_percentage = new_data_fraction  

    
    if subset_percentage >= 1.0 and early_stop_count==10:
        print("Training on all data subsets complete.")
        break  


coral_model.load_state_dict(torch.load('./best_model.ckpt'))
test(coral_model, test_dataloader, "test")
