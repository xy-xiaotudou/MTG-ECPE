import re

import torch
import torch.nn as nn
import argparse
import os
import pytorch_lightning as pl
import logging
import time

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import seed_everything
from data_utils import SINADataset, write_results_to_log, read_sina_line_examples_from_file
from data_utils import NTCIRDataset, read_ntcir_line_examples_from_file
from eval_utils import compute_scores, compute_scores_emotask
from tqdm import tqdm
from os.path import join

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='ecpe', type=str, required=True,
                        help="The name of the task, selected from: [ecpe]")
    parser.add_argument("--dataset", default='eca_cn_10', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16, eca_ch]")
    parser.add_argument("--model_name_or_path", default='google/mt5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='multi_task', type=str, required=True,
                        help="The way to construct target sentence, selected from: [multi_task, only_ecpe, ecpe_wo_label, wo_all_label]")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run direct eval on the dev/test set.")
    parser.add_argument("--mode", default=False, type=bool, help="emotion task and ecpe task interaction or not")
    parser.add_argument("--formatted", default=False, type=bool, help="clause with number id")
    # Other parameters
    parser.add_argument("--max_seq_length", default=30, type=int)  ## ch:128 en:30
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)  ## eng:0.005
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=2024, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, fold_id, type_path, args):
    if args.dataset == 'eca_cn_10' or args.dataset == 'eca_cn_20':
        return SINADataset(tokenizer=tokenizer, fold_id=fold_id, data_type=type_path, task=args.task,
                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
    elif args.dataset == 'eca_eng':
        return NTCIRDataset(tokenizer=tokenizer, fold_id=fold_id, data_type=type_path, task=args.task,
                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)


class shareT5Encoder(pl.LightningModule):
    def __init__(self, hparams, fold_id):
        super(shareT5Encoder, self).__init__()
        self.fold_id = fold_id
        self.hparams = hparams
        self.base_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.mode = hparams.mode
        self.shared_encoder = self.base_model.get_encoder()  # 初始化独立解码器

        # 初始化情感子句提取解码器
        self.emo_decoder = self.base_model.get_decoder()
        self.emo_decoder.resize_token_embeddings(len(self.tokenizer))

        # 假设词表大小为 vocab_size，hidden_size 为解码器隐状态维度
        vocab_size = len(self.tokenizer)
        hidden_size = self.emo_decoder.config.d_model  # T5 模型的隐状态维度

        # 初始化一个线性投影层
        self.projection_layer = nn.Linear(hidden_size, vocab_size)

        # 初始化情感原因对提取解码器
        self.cause_decoder = self.base_model.get_decoder()
        self.cause_decoder.resize_token_embeddings(len(self.tokenizer))

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # print('Running _step...')
        total_loss = 0.0
        predicted_emo_clauses = []

        task_types = batch.get('task_type')  # 获取任务类型
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
        target_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        for i, task in enumerate(task_types):
            loss_emo_task, loss_ecpe_task = 0.0, 0.0
            input_text = input_texts[i]
            target_text = target_texts[i]
            if self.hparams.dataset == 'eca_eng':
                # 根据任务类型生成描述文本并获取任务编码
                if task == 'emotion clause extraction':
                    task_description = "emotion clause extraction:"
                elif task == 'emotion cause pair extraction':
                    task_description = "emotion cause pair extraction:"
                else:
                    raise ValueError(f"Unknown task type for sample {i}: {input_text}")
            else:
                # 根据任务类型生成描述文本并获取任务编码
                if task == '情感子句提取':
                    task_description = "情感子句提取:"
                elif task == '情感原因对提取':
                    task_description = "情感原因对提取:"
                else:
                    raise ValueError(f"Unknown task type for sample {i}: {input_text}")

            print(f'Task {i}: {task}')

            # 将任务描述显式添加到输入文本的开头
            augmented_input_text = f"{task_description} {input_text}"

            # 使用拼接后的文本生成模型输入
            tokenizer_output = self.tokenizer(augmented_input_text, max_length=512,
                                              padding='max_length', truncation=True, return_tensors="pt")

            inputs_padded = tokenizer_output["input_ids"].to(self.device)
            attention_mask = tokenizer_output["attention_mask"].to(self.device)

            task_batch = {
                "source_ids": inputs_padded,
                "source_mask": attention_mask,
                "target_ids": batch["target_ids"][i].unsqueeze(0).to(self.device),
                "target_mask": batch["target_mask"][i].unsqueeze(0).to(self.device),
            }
            #
            print('--Embedded Source--', self.tokenizer.decode(task_batch["source_ids"][0], skip_special_tokens=True))
            print('--Target--', self.tokenizer.decode(task_batch["target_ids"][0], skip_special_tokens=True))

            # 编码器部分共享
            encoder_outputs = self.shared_encoder(
                input_ids=task_batch["source_ids"],
                attention_mask=task_batch["source_mask"]
            )
            # print('encoder_outputs:', encoder_outputs.last_hidden_state.shape)

            # 处理情感子句提取任务
            if task == '情感子句提取' or task == 'emotion clause extraction':
                emo_decoder_outputs = self.emo_decoder(
                    input_ids=task_batch["target_ids"],
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=task_batch["source_mask"]
                )
                # print('emo_decoder_outputs:', emo_decoder_outputs)
                # logits = self.projection_layer(emo_decoder_outputs.last_hidden_state)
                loss_emo_task = self.compute_loss(emo_decoder_outputs, task_batch["target_ids"], task_batch["source_mask"])
                # # 解码生成结果
                # generated_text = self.tokenizer.decode(
                #     emo_decoder_outputs.logits.argmax(dim=-1)[0],
                #     skip_special_tokens=True
                # )
                # predicted_emo_clauses.append(generated_text)  # 保存情感子句

            # 处理情感原因对提取任务
            elif task == '情感原因对提取' or task == 'emotion cause pair extraction':
                cause_decoder_outputs = self.cause_decoder(
                    input_ids=task_batch["target_ids"],
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=task_batch["source_mask"]
                )
                loss_ecpe_task = self.compute_loss(cause_decoder_outputs, task_batch["target_ids"], task_batch["source_mask"])

            total_loss += 0.2 * loss_emo_task + 0.8 * loss_ecpe_task

        return total_loss

    def compute_loss(self, decoder_outputs, target_ids, target_mask):
        """
        计算损失，使用CrossEntropyLoss
        """
        # 投影到词表分布
        logits = self.projection_layer(decoder_outputs.last_hidden_state)
        # print('logits:', logits.shape)
        target_ids = target_ids.view(-1)  # 形状: [batch_size * sequence_length]
        # print('target_ids:', target_ids.shape)
        target_mask = target_mask.view(-1)  # 形状: [batch_size * sequence_length]
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids)
        # print('loss:', loss)
        # 仅计算有效 token 的损失
        loss = (loss * target_mask).sum() / target_mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        total_loss = self._step(batch)
        tensorboard_logs = {"train_loss": total_loss}
        return {"loss": total_loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        total_loss = self._step(batch)
        return {"val_loss": total_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        no_decay = ["bias", "LayerNorm.weight"]
        # 收集所有参数，防止重复
        unique_params = set()

        def filter_params(named_params, no_decay, unique_params):
            """过滤参数，确保不重复"""
            params_with_decay = []
            params_without_decay = []
            for n, p in named_params:
                if p in unique_params:
                    continue  # 跳过已经添加的参数
                if any(nd in n for nd in no_decay):
                    params_without_decay.append(p)
                else:
                    params_with_decay.append(p)
                unique_params.add(p)  # 标记参数为已添加
            return [
                {"params": params_with_decay, "weight_decay": self.hparams.weight_decay},
                {"params": params_without_decay, "weight_decay": 0.0},
            ]

        # 获取共享编码器的参数
        encoder_params = filter_params(self.shared_encoder.named_parameters(), no_decay, unique_params)

        # 获取原因解码器的参数
        cause_decoder_params = filter_params(self.cause_decoder.named_parameters(), no_decay, unique_params)

        # 获取情感编码器的参数
        emotion_decoder_params = filter_params(self.emo_decoder.named_parameters(), no_decay, unique_params)

        # 合并参数
        optimizer_grouped_parameters = encoder_params + cause_decoder_params + emotion_decoder_params

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def total_training_steps(self):
        dataset_size = len(self.train_dataloader().dataset)
        steps_per_epoch = dataset_size // self.hparams.train_batch_size
        return steps_per_epoch * self.hparams.num_train_epochs

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, fold_id=self.fold_id, type_path="train",
                                    args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=0)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, fold_id=self.fold_id, type_path="test", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=0)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def generate_text_share(task_name, input_text, tokenizer, model, device, target_ids, target_mask, max_length=512):
    """
    Encode input text and generate output text using the model.
    """
    # print('input_text：', input_text)
    # 使用拼接后的文本生成模型输入
    tokenizer_output = tokenizer(input_text, max_length=512,
                                      padding='max_length', truncation=True, return_tensors="pt")
    inputs_padded = tokenizer_output["input_ids"].to(device)
    attention_mask = tokenizer_output["attention_mask"].to(device)

    task_batch = {
        "source_ids": inputs_padded,
        "source_mask": attention_mask,
        "target_ids": target_ids.unsqueeze(0).to(device),
        "target_mask": target_mask.unsqueeze(0).to(device),
    }

    # print('--Embedded Source--', tokenizer.decode(task_batch["source_ids"][0], skip_special_tokens=True))
    # print('--Target--', tokenizer.decode(task_batch["target_ids"][0], skip_special_tokens=True))

    # Shared encoder
    encoder_outputs = model.shared_encoder(input_ids=task_batch["source_ids"], attention_mask=task_batch["source_mask"])
    # 处理情感子句提取任务
    if task_name == '情感子句提取' or task_name == 'emotion clause extraction':
        emo_decoder_outputs = model.emo_decoder(
            input_ids=task_batch["target_ids"],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=task_batch["source_mask"]
        )
        # 投影到词表分布
        logits = model.projection_layer(emo_decoder_outputs.last_hidden_state)
        # print('logits:', logits.shape)
        # tokenizer.eos_token_id 是结束标记 ID
        eos_token_id = tokenizer.eos_token_id
        # max_length = 50  # 最大解码长度
        predicted_ids = []
        for seq_logits in logits:
            seq = []
            for token_logits in seq_logits:
                token_id = torch.argmax(token_logits).item()
                if token_id == eos_token_id:
                    break
                seq.append(token_id)
                if len(seq) >= max_length:
                    break
            predicted_ids.append(seq)
        # 将生成的 token IDs 转化为文本
        decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_ids]
        # print('decoded_texts:', decoded_texts)

    # 处理情感原因对提取任务
    elif task_name == '情感原因对提取' or task_name == 'emotion cause pair extraction':
        cause_decoder_outputs = model.cause_decoder(
            input_ids=task_batch["target_ids"],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=task_batch["source_mask"]
        )
        logits = model.projection_layer(cause_decoder_outputs.last_hidden_state)
        # predicted_ids = torch.argmax(logits, dim=-1)  # 在 vocab_size 维度上选择最高分的 token ID
        eos_token_id = tokenizer.eos_token_id
        # max_length = 50  # 最大解码长度
        predicted_ids = []
        for seq_logits in logits:
            seq = []
            for token_logits in seq_logits:
                token_id = torch.argmax(token_logits).item()
                if token_id == eos_token_id:
                    break
                seq.append(token_id)
                if len(seq) >= max_length:
                    break
            predicted_ids.append(seq)
        # 将生成的 token IDs 转化为文本
        decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_ids]
        # print('decoded_texts:', decoded_texts)
    return decoded_texts


def evaluate_share(args, tokenizer, data_loader, model, paradigm, task, dataset, sents):
    """
    Compute scores given the predictions and gold labels
    """
    # device = torch.device(f'cuda:{args.n_gpu}')
    device = torch.device(f'cuda:{args.n_gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    emo_outputs, emo_targets, ecpe_outputs, ecpe_targets = [], [], [], []

    for batch in tqdm(data_loader):
        # make sure task type
        task_type = batch.get('task_type')
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
        target_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        batch_outputs = []
        # print('mode:', args.mode)
        for i, task_name in enumerate(task_type):
            if task_name == '情感子句提取' or task_name == 'emotion clause extraction':
                task_input_text = task_name + ':' + input_texts[i]
                target_ids = batch['target_ids'][i]
                target_mask = batch["target_mask"][i]
                output = generate_text_share(task_name, task_input_text, tokenizer, model, device, target_ids, target_mask)
                batch_outputs.append(output[0])
                emo_outputs.append(output[0])
                emo_targets.append(target_texts[i])
            elif task_name == '情感原因对提取' or task_name == 'emotion cause pair extraction':
                ### self.mode 没有用
                if args.mode == True:
                    # Step 1: first process emotion task
                    emo_task_input_text = f"情感子句提取:{input_texts[i]}"
                    # print('emo_task_input_text:', emo_task_input_text)
                    extracted_emotion_clauses = generate_text_share('情感子句提取', emo_task_input_text, tokenizer, model, device, target_ids, target_mask)

                    # Step 2: ECPE with emotion results
                    ecpe_task_input = (
                        f"{task_name}:{input_texts[i]}. 文档中的情感子句: {extracted_emotion_clauses}"
                    )
                    output = generate_text_share(task_name, ecpe_task_input, tokenizer, model, device)
                    batch_outputs.append(output)
                    ecpe_outputs.append(output)
                    ecpe_targets.append(target_texts[i])
                elif args.mode == False:
                    task_input_text = task_name + ':' + input_texts[i]
                    target_ids = batch['target_ids'][i]
                    target_mask = batch["target_mask"][i]
                    output = generate_text_share(task_name, task_input_text, tokenizer, model, device, target_ids, target_mask)
                    batch_outputs.append(output[0])
                    ecpe_outputs.append(output[0])
                    ecpe_targets.append(target_texts[i])
                    # print('output[0], target_texts[i]:', output[0], target_texts[i])

    print('paradigm:', paradigm)
    if paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                    'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
        raw_scores_emo, fixed_scores_emo, all_labels_emo, all_predictions_emo, all_predictions_fixed_emo = (
                compute_scores_emotask(emo_outputs, emo_targets, sents, paradigm, task, dataset))
        results_ee = {'raw_scores_emo': raw_scores_emo, 'fixed_scores_emo': fixed_scores_emo,
                          'labels': all_labels_emo,
                          'preds': all_predictions_emo, 'preds_fixed': all_predictions_fixed_emo}
        print('Emotion Task results_ee:', results_ee)

    print('ecpe_targets:', ecpe_targets)
    print('ecpe_outputs:', ecpe_outputs)
    raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce, \
            all_labels, all_predictions, all_predictions_fixed =\
            (compute_scores(ecpe_outputs, ecpe_targets, sents, paradigm, task, dataset))
    results_ecpe = {'raw_scores_ecpe': raw_scores_ecpe, 'fixed_scores_ecpe': fixed_scores_ecpe,
                        'raw_scores_ee': raw_scores_ee, 'fixed_scores_ee': fixed_scores_ee,
                        'raw_scores_ce': raw_scores_ce, 'fixed_scores_ce': fixed_scores_ce,
                        'labels': all_labels,
                        'preds': all_predictions, 'preds_fixed': all_predictions_fixed}
    # print('ECPE Task results_ecpe:', results_ecpe)
    if paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                    'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
        return results_ee, results_ecpe
    else:
        return results_ecpe


def initialize_scores():
    return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def main():
    # initialization
    print('-' * 10, 'initialization', '-' * 10)
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "=" * 30, "\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    seed_everything(args.seed)
    # 初始化 T5 模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    if args.dataset == 'eca_cn_20':
        fold_num = 21
    else:
        fold_num = 11

    for fold_id in range(1, fold_num):
        print('-' * 15, 'fold{}'.format(fold_id), '-' * 15)
        print('args.do_train, do_eval, formatted:', args.do_train, args.do_eval, args.formatted)
        print(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:", args.dataset)
        if args.dataset == 'eca_cn_10':
            dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
        elif args.dataset == 'eca_cn_20':
            dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                        paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
        elif args.dataset == 'eca_eng':
            dataset = NTCIRDataset(tokenizer=tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                  paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
        data_sample = dataset[0]  # a random data sample
        print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
        print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))


        if args.do_train:
            print("\n****** Conduct Training ******")
            # model = T5FineTuner(args, fold_id=fold_id)
            model = shareT5Encoder(args, fold_id=fold_id)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=f"{args.output_dir}/fold_{fold_id}/ckt_{{epoch}}_{{val_loss:.2f}}",
                prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
            )

            # prepare for trainer
            train_params = dict(
                default_root_dir=args.output_dir,
                accumulate_grad_batches=args.gradient_accumulation_steps,
                # gpus=args.n_gpu,
                gradient_clip_val=1.0,
                amp_level='O1',
                max_epochs=args.num_train_epochs,
                checkpoint_callback=checkpoint_callback,
                callbacks=[LoggingCallback()],
            )

            trainer = pl.Trainer(**train_params)
            trainer.fit(model)

            print("Finish training and saving the model!")

        if args.do_eval:
            print("\n****** Conduct Evaluating ******")
            # model = T5FineTuner(args)
            dev_results_ecp, test_results_ecp = {}, {}
            best_f1, best_checkpoint, best_epoch = -999999.0, None, None
            all_checkpoints, all_epochs = [], []

            saved_model_dir = args.output_dir

            # print('saved_model_dir:', saved_model_dir)
            current_fold_dir = 'fold_'+str(fold_id)
            current_fold_path = os.path.join(saved_model_dir, current_fold_dir)
            for f in os.listdir(current_fold_path):
                file_name = os.path.join(current_fold_path, f)
                # print('file_name:', file_name)
                # 检查文件名中是否包含 'cktepoch'
                if 'cktckt_epoch' in file_name:
                    all_checkpoints.append(file_name)

            # conduct some selection (or not)
            print(f"We will perform validation on the following checkpoints: {all_checkpoints}")
            DEV_FILE = 'fold%s_val.txt'
            TEST_FILE = 'fold%s_test.txt'

            # load dev and test datasets
            if args.dataset == 'eca_cn_10':
                dev_data_path = join('data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, dev_doc_sents, dev_doc_format_sents, _, _, _, _, _ = (read_sina_line_examples_from_file(dev_data_path))
                dev_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                          paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=0)

                test_data_path = join('data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, test_doc_sents, test_doc_format_sents, _, _, _, _, _ = (read_sina_line_examples_from_file(test_data_path))
                test_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
            elif args.dataset == 'eca_cn_20':
                dev_data_path = join('data/' + args.task + '/' + args.dataset, DEV_FILE % fold_id)
                # print('dev_data_path:', dev_data_path)
                _, _, dev_doc_sents, dev_doc_format_sents, _, _, _, _, _ = (read_sina_line_examples_from_file(dev_data_path))
                dev_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='val', task=args.task,
                                          paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=0)

                test_data_path = join('data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, test_doc_sents, test_doc_format_sents, _, _, _, _, _ = (read_sina_line_examples_from_file(test_data_path))
                test_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
            elif args.dataset == 'eca_eng':
                dev_data_path = join('data/' + args.task + '/' + args.dataset, DEV_FILE % fold_id)
                _, _, dev_doc_sents, dev_doc_format_sents, _, _, _, _, _ = (read_ntcir_line_examples_from_file(dev_data_path))
                print('dev_doc_sents:', dev_doc_sents)
                dev_dataset = NTCIRDataset(tokenizer, fold_id=fold_id, data_type='val', task=args.task,
                                          paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=0)

                test_data_path = join('data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, test_doc_sents, test_doc_format_sents, _, _, _, _, _ = (read_ntcir_line_examples_from_file(test_data_path))
                test_dataset = NTCIRDataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
                test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

            for checkpoint in all_checkpoints:
                epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
                if epoch != ' ':
                    all_epochs.append(epoch)
                    # reload the model and conduct inference
                    print(f"\nLoad the trained model from {checkpoint}...")
                    model_ckpt = torch.load(checkpoint)
                    model = shareT5Encoder(model_ckpt['hyper_parameters'], fold_id=fold_id)
                    model.load_state_dict(model_ckpt['state_dict'])

                    if args.formatted == True:
                        if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                             'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                            dev_result_ee, dev_result_ecpe = evaluate_share(args, tokenizer, dev_loader, model,
                                                                            args.paradigm, args.task, args.dataset,
                                                                            dev_doc_format_sents)
                        else:
                            dev_result_ecpe = evaluate_share(args, tokenizer, dev_loader, model,
                                                             args.paradigm, args.task, dev_doc_format_sents)
                    else:
                        if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                             'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                            dev_result_ee, dev_result_ecpe = evaluate_share(args, tokenizer, dev_loader, model,
                                                                            args.paradigm, args.task, args.dataset,
                                                                            dev_doc_sents)
                        else:
                            dev_result_ecpe = evaluate_share(args, tokenizer, dev_loader, model,
                                                             args.paradigm, args.task, args.dataset, dev_doc_sents)

                            dev_result_ee = {
                                'raw_scores_emo': initialize_scores(),
                                'fixed_scores_emo': initialize_scores()
                            }

                    # write to file
                    log_file_path = f"results_log/{args.task}-{args.dataset}-{args.paradigm}.txt"
                    local_time = time.asctime(time.localtime(time.time()))
                    exp_settings = f"{args.task} on {args.dataset} under {args.paradigm} epoch {epoch}; " \
                                   f"fold_id = {fold_id}," \
                                   f"Train bs={args.train_batch_size}, " \
                                   f"num_epochs = {args.num_train_epochs}"

                    dev_exp_results = f"Raw ecpe P={dev_result_ecpe['raw_scores_ecpe']['precision']:.4f}, R={dev_result_ecpe['raw_scores_ecpe']['recall']:.4f}, F1 = {dev_result_ecpe['raw_scores_ecpe']['f1']:.4f}, \n" \
                                      f"Fixed ecpe P={dev_result_ecpe['fixed_scores_ecpe']['precision']:.4f}, R={dev_result_ecpe['fixed_scores_ecpe']['recall']:.4f}, F1 = {dev_result_ecpe['fixed_scores_ecpe']['f1']:.4f}, \n" \
                                      f"Raw ee P={dev_result_ecpe['raw_scores_ee']['precision']:.4f}, R={dev_result_ecpe['raw_scores_ee']['recall']:.4f}, F1 = {dev_result_ecpe['raw_scores_ee']['f1']:.4f}, \n" \
                                      f"Fixed ee P={dev_result_ecpe['fixed_scores_ee']['precision']:.4f}, R={dev_result_ecpe['fixed_scores_ee']['recall']:.4f}, F1 = {dev_result_ecpe['fixed_scores_ee']['f1']:.4f}, \n" \
                                      f"Raw ce P={dev_result_ecpe['raw_scores_ce']['precision']:.4f}, R={dev_result_ecpe['raw_scores_ce']['recall']:.4f}, F1 = {dev_result_ecpe['raw_scores_ce']['f1']:.4f}, \n" \
                                      f"Fixed ce P={dev_result_ecpe['fixed_scores_ce']['precision']:.4f}, R={dev_result_ecpe['fixed_scores_ce']['recall']:.4f}, F1 = {dev_result_ecpe['fixed_scores_ce']['f1']:.4f}, \n" \
                                      f"Raw emo P={dev_result_ee['raw_scores_emo']['precision']:.4f}, R={dev_result_ee['raw_scores_emo']['recall']:.4f}, F1 = {dev_result_ee['raw_scores_emo']['f1']:.4f}, \n" \
                                      f"Fixed emo P={dev_result_ee['fixed_scores_emo']['precision']:.4f}, R={dev_result_ee['fixed_scores_emo']['recall']:.4f}, F1 = {dev_result_ee['fixed_scores_emo']['f1']:.4f} \n"
                    log_dev_str = f'============================dev results================================\n'
                    log_dev_str += f"{local_time}\n{exp_settings}\n{dev_exp_results}\n\n"
                    with open(log_file_path, "a+") as f:
                        f.write(log_dev_str)

                    if dev_result_ecpe['raw_scores_ecpe']['f1'] > best_f1:
                        best_f1 = dev_result_ecpe['raw_scores_ecpe']['f1']
                        best_checkpoint = checkpoint
                        best_epoch = epoch

                    # add the global step to the name of these metrics for recording
                    # 'f1' --> 'f1_1000'
                    dev_result_ecpe = dict((k + '_{}'.format(epoch), v) for k, v in dev_result_ecpe.items())
                    dev_results_ecp.update(dev_result_ecpe)

                    if args.formatted == True:
                        if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                             'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                            test_result_ee, test_result_ecpe = evaluate_share(args, tokenizer, test_loader, model,
                                                                              args.paradigm, args.task, args.dataset,
                                                                              test_doc_format_sents)
                        else:
                            test_result_ecpe = evaluate_share(args, tokenizer, test_loader, model,
                                                              args.paradigm, args.task, args.dataset,
                                                              test_doc_format_sents)
                            test_result_ee = {
                                'raw_scores_emo': initialize_scores(),
                                'fixed_scores_emo': initialize_scores()
                            }
                    else:
                        if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                             'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                            test_result_ee, test_result_ecpe = evaluate_share(args, tokenizer, test_loader, model,
                                                                              args.paradigm, args.task, args.dataset,
                                                                              test_doc_sents)
                        else:
                            test_result_ecpe = evaluate_share(args, tokenizer, test_loader, model,
                                                              args.paradigm, args.task, args.dataset, test_doc_sents)
                            test_result_ee = {
                                'raw_scores_emo': initialize_scores(),
                                'fixed_scores_emo': initialize_scores()
                            }
                    # write to file
                    local_time = time.asctime(time.localtime(time.time()))

                    text_exp_results = f"Raw ecpe P={test_result_ecpe['raw_scores_ecpe']['precision']:.4f}, R={test_result_ecpe['raw_scores_ecpe']['recall']:.4f}, F1 = {test_result_ecpe['raw_scores_ecpe']['f1']:.4f}, \n" \
                                       f"Fixed ecpe P={test_result_ecpe['fixed_scores_ecpe']['precision']:.4f}, R={test_result_ecpe['fixed_scores_ecpe']['recall']:.4f}, F1 = {test_result_ecpe['fixed_scores_ecpe']['f1']:.4f}, \n" \
                                       f"Raw ee P={test_result_ecpe['raw_scores_ee']['precision']:.4f}, R={test_result_ecpe['raw_scores_ee']['recall']:.4f}, F1 = {test_result_ecpe['raw_scores_ee']['f1']:.4f}, \n" \
                                       f"Fixed ee P={test_result_ecpe['fixed_scores_ee']['precision']:.4f}, R={test_result_ecpe['fixed_scores_ee']['recall']:.4f}, F1 = {test_result_ecpe['fixed_scores_ee']['f1']:.4f}, \n" \
                                       f"Raw ce P={test_result_ecpe['raw_scores_ce']['precision']:.4f}, R={test_result_ecpe['raw_scores_ce']['recall']:.4f}, F1 = {test_result_ecpe['raw_scores_ce']['f1']:.4f}, \n" \
                                       f"Fixed ce P={test_result_ecpe['fixed_scores_ce']['precision']:.4f}, R={test_result_ecpe['fixed_scores_ce']['recall']:.4f}, F1 = {test_result_ecpe['fixed_scores_ce']['f1']:.4f}, \n" \
                                       f"Raw emo P={test_result_ee['raw_scores_emo']['precision']:.4f}, R={test_result_ee['raw_scores_emo']['recall']:.4f}, F1 = {test_result_ee['raw_scores_emo']['f1']:.4f}, \n" \
                                       f"Fixed emo P={test_result_ee['fixed_scores_emo']['precision']:.4f}, R={test_result_ee['fixed_scores_emo']['recall']:.4f}, F1 = {test_result_ee['fixed_scores_emo']['f1']:.4f} \n "
                    log_text_str = f'===========================text results=================================\n'
                    log_text_str += f"{local_time}\n{exp_settings}\n{text_exp_results}\n\n"
                    with open(log_file_path, "a+") as f:
                        f.write(log_text_str)

                    test_result_ecpe = dict((k + '_{}'.format(epoch), v) for k, v in test_result_ecpe.items())
                    test_results_ecp.update(test_result_ecpe)

            # print test results over last few steps
            print(f"\n\nThe best checkpoint is {best_checkpoint}")
            best_step_metric = f"raw_scores_ecpe_{best_epoch}"
            print(f"ECPE Task F1 scores on test set: {test_results_ecp[best_step_metric]['f1']:.4f}")

            print("\n* Results *:  Dev  /  Test  \n")
            metric_names = ['f1', 'precision', 'recall']
            for epoch in all_epochs:
                print(f"Epoch-{epoch}:")
                task_name = f'raw_scores_ecpe_{epoch}'
                for name in metric_names:
                    name_step = f'{name}'
                    print(
                        f"{name:<10}: {dev_results_ecp[task_name][name_step]:.4f} / {test_results_ecp[task_name][name_step]:.4f}",
                        sep='  ')
                print()
            results_log_dir = './results_log'
            if not os.path.exists(results_log_dir):
                os.mkdir(results_log_dir)
            log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}-{args.paradigm}.txt"
            write_results_to_log(log_file_path, test_results_ecp[best_step_metric]['f1'], args,
                                 dev_results_ecp, test_results_ecp, all_epochs)

        # evaluation process
        if args.do_direct_eval:
            print("\n****** Conduct Evaluating with the last state ******")

            TEST_FILE = 'fold%s_test.txt'
            if args.dataset == 'eca_cn_10' or args.dataset == 'eca_cn_20':
                data_path = join('./data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, test_doc_sents, test_doc_format_sents, _, _, _, _, _ = read_sina_line_examples_from_file(data_path)

                test_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
            elif args.dataset == 'eca_eng':
                data_path = join('./data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
                _, _, test_doc_sents, test_doc_format_sents, _, _, _, _, _ = read_ntcir_line_examples_from_file(data_path)
                test_dataset = NTCIRDataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                            paradigm=args.paradigm, data_dir=args.dataset, clauseid=args.formatted)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

            if args.task in ['ecpe']:
                if args.formatted == False:
                    if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                         'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                        test_results_ee, test_results_ecpe = \
                            evaluate_share(args, tokenizer, test_loader, model, args.paradigm, args.task, args.dataset,
                                           test_doc_sents)
                    else:
                        test_results_ecpe = \
                            evaluate_share(args, tokenizer, test_loader, model, args.paradigm, args.task, args.dataset,
                                           test_doc_sents)
                        test_results_ee = {
                            'raw_scores_emo': initialize_scores(),
                            'fixed_scores_emo': initialize_scores()
                        }
                elif args.formatted == True:
                    if args.paradigm in ['multi_task', 'wo_all_label', 'multi_task_mee', 'wo_clause_types',
                                         'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                        test_results_ee, test_results_ecpe = \
                            evaluate_share(args, tokenizer, test_loader, model, args.paradigm, args.task, args.dataset,
                                           test_doc_format_sents)
                    else:
                        test_results_ecpe = \
                            evaluate_share(args, tokenizer, test_loader, model, args.paradigm, args.task, args.dataset,
                                           test_doc_format_sents)
                        test_results_ee = {
                            'raw_scores_emo': initialize_scores(),
                            'fixed_scores_emo': initialize_scores()
                        }
                # write to file
                log_file_path = f"results_log/{args.task}-{args.dataset}-{args.paradigm}.txt"
                local_time = time.asctime(time.localtime(time.time()))
                exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; " \
                               f"fold_id = {fold_id}," \
                               f"Train bs={args.train_batch_size}, " \
                               f"num_epochs = {args.num_train_epochs}"
                exp_results = f"Raw ecpe P={test_results_ecpe['raw_scores_ecpe']['precision']:.4f}, R={test_results_ecpe['raw_scores_ecpe']['recall']:.4f}, F1 = {test_results_ecpe['raw_scores_ecpe']['f1']:.4f}, \n" \
                              f"Fixed ecpe P={test_results_ecpe['fixed_scores_ecpe']['precision']:.4f}, R={test_results_ecpe['fixed_scores_ecpe']['recall']:.4f}, F1 = {test_results_ecpe['fixed_scores_ecpe']['f1']:.4f}, \n" \
                              f"Raw ee P={test_results_ecpe['raw_scores_ee']['precision']:.4f}, R={test_results_ecpe['raw_scores_ee']['recall']:.4f}, F1 = {test_results_ecpe['raw_scores_ee']['f1']:.4f}, \n" \
                              f"Fixed ee P={test_results_ecpe['fixed_scores_ee']['precision']:.4f}, R={test_results_ecpe['fixed_scores_ee']['recall']:.4f}, F1 = {test_results_ecpe['fixed_scores_ee']['f1']:.4f}, \n" \
                              f"Raw ce P={test_results_ecpe['raw_scores_ce']['precision']:.4f}, R={test_results_ecpe['raw_scores_ce']['recall']:.4f}, F1 = {test_results_ecpe['raw_scores_ce']['f1']:.4f}, \n" \
                              f"Fixed ce P={test_results_ecpe['fixed_scores_ce']['precision']:.4f}, R={test_results_ecpe['fixed_scores_ce']['recall']:.4f}, F1 = {test_results_ecpe['fixed_scores_ce']['f1']:.4f}, \n" \
                              f"Raw emo P={test_results_ee['raw_scores_emo']['precision']:.4f}, R={test_results_ee['raw_scores_emo']['recall']:.4f}, F1 = {test_results_ee['raw_scores_emo']['f1']:.4f}, \n" \
                              f"Fixed emo P={test_results_ee['fixed_scores_emo']['precision']:.4f}, R={test_results_ee['fixed_scores_emo']['recall']:.4f}, F1 = {test_results_ee['fixed_scores_emo']['f1']:.4f} \n "
                log_str = f'============================direct================================\n'
                log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
                with open(log_file_path, "a+") as f:
                    f.write(log_str)


if __name__ == '__main__':
    main()