import time

from os.path import join
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 初始化 T5 模型和分词器
model_name = "google-t5/t5-small"  # 可以选择 "t5-base" 或更大的模型
# model_name = "./model/t5-small"  # 可以选择 "t5-base" 或更大的模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

TRAIN_FILE = 'fold%s_train.txt'
VALID_FILE = 'fold%s_val.txt'
DEV_FILE = 'fold%s_val.txt'
TEST_FILE = 'fold%s_test.txt'


def read_ntcir_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, \
    doc_cau_list, doc_emocat_list, doc_emokeyword_list = [], [], [], [], [], [], [], []
    doc_format_sents_list = []
    inputFile = open(data_path, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = line[0]
        doc_id_list.append(doc_id)
        doc_len = int(line[1])
        doc_len_list.append(doc_len)

        ecps = eval('[' + inputFile.readline().strip() + ']')
        doc_ecps_list.append(ecps)
        emo_l, cau_l = [], []
        for pi in ecps:
            if pi[0] not in emo_l:
                emo_l.append(pi[0])
            if pi[1] not in cau_l:
                cau_l.append(pi[1])
        doc_emo_list.append(emo_l)
        doc_cau_list.append(cau_l)

        contents_l, doc_emocat_l, doc_emokeyword_l = [], [], []
        f_contents_l = []
        for i in range(doc_len):
            clause_line = inputFile.readline().strip().split(',')
            emo_cat = clause_line[1]
            emo_keyword = clause_line[2]
            if emo_cat != 'null':
                doc_emocat_l.append(emo_cat)
                doc_emokeyword_l.append(emo_keyword)
            content = clause_line[-1] #.replace(' ', '')
            if content.endswith(' .'):
                content = content.replace(' .', '.')
            formatted_content = 'c%d:%s' % (i+1, content)
            contents_l.append(content)
            f_contents_l.append(formatted_content)
        doc_emocat_list.append(doc_emocat_l)
        doc_sents_list.append(contents_l)
        doc_format_sents_list.append(f_contents_l)
        doc_emokeyword_list.append(doc_emokeyword_l)
    return doc_id_list, doc_len_list, doc_sents_list, doc_format_sents_list, \
           doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list, doc_emokeyword_list


def get_ntcir_transformed_io(data_path, paradigm, task, formatted_clauseid):
    """
    doc_emos: emotion clause' id
    doc_caus: cause clause' id
    doc_emocat: emotion clause's sentiment type
    doc_emokeyword:
    """
    doc_ids, doc_lens, doc_sents, doc_format_sents, \
    doc_ecps, doc_emos, doc_caus, doc_emocat, doc_emokeyword = read_ntcir_line_examples_from_file(data_path)

    data = []
    if paradigm == 'multi_task':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            # input_text = ','.join(doc_sents[id])
            input_text = ','.join(doc_format_sents[id])

            ## EE sub-task
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"emotion clause:{doc_format_sents[id][doc_emos[id][eid]-1]},emotion type:{doc_emocat[id][eid]},clue words:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
                ## ---set2--- 情感子句：XXX,情感类别：XXX
                # ee_target_text = ";".join([f"emotion clause:{doc_sents[id][doc_emos[id][eid]-1]},emotion type:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])
                ## ---set3--- 情感子句：ci：XXX,情感类别：XXX
                # ee_target_text = ";".join([f"emotion clause:{doc_sents[id][doc_emos[id][eid]-1]},emotion type:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"emotion clause:{doc_sents[id][doc_emos[id][eid]-1]},emotion type:{doc_emocat[id][eid]},clue words:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "emotion clause extraction",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"emotion clause:{emotion}, cause clause:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "emotion cause pair extraction",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)

    return data


def read_sina_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, \
    doc_cau_list, doc_emocat_list, doc_emokeyword_list = [], [], [], [], [], [], [], []
    doc_format_sents_list = []
    inputFile = open(data_path, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = line[0]
        doc_id_list.append(doc_id)
        doc_len = int(line[1])
        doc_len_list.append(doc_len)

        ecps = eval('[' + inputFile.readline().strip() + ']')
        doc_ecps_list.append(ecps)
        emo_l, cau_l = [], []
        for pi in ecps:
            if pi[0] not in emo_l:
                emo_l.append(pi[0])
            if pi[1] not in cau_l:
                cau_l.append(pi[1])
        doc_emo_list.append(emo_l)
        doc_cau_list.append(cau_l)

        contents_l, doc_emocat_l, doc_emokeyword_l = [], [], []
        f_contents_l = []
        for i in range(doc_len):
            clause_line = inputFile.readline().strip().split(',')
            emo_cat = clause_line[1]
            emo_keyword = clause_line[2]
            if emo_cat != 'null':
                doc_emocat_l.append(emo_cat)
                doc_emokeyword_l.append(emo_keyword)
            content = clause_line[-1].replace(' ', '')
            formatted_content = 'c%d:%s' % (i+1, content)
            contents_l.append(content)
            f_contents_l.append(formatted_content)
        doc_emocat_list.append(doc_emocat_l)
        doc_sents_list.append(contents_l)
        doc_format_sents_list.append(f_contents_l)
        doc_emokeyword_list.append(doc_emokeyword_l)
    return doc_id_list, doc_len_list, doc_sents_list, doc_format_sents_list, \
           doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list, doc_emokeyword_list


def get_sina_transformed_io(data_path, paradigm, task, formatted_clauseid):
    """
    doc_emos: emotion clause' id
    doc_caus: cause clause' id
    doc_emocat: emotion clause's sentiment type
    doc_emokeyword:
    """
    doc_ids, doc_lens, doc_sents, doc_format_sents, \
    doc_ecps, doc_emos, doc_caus, doc_emocat, doc_emokeyword = read_sina_line_examples_from_file(data_path)

    data = []
    if paradigm == 'multi_task':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
            ## ---set2--- 情感子句：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])
            ## ---set3--- 情感子句：ci：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'multi_task_mee':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                # ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
                ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid]-1]}" for eid in range(len(doc_emos[id]))])

            else:
                # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
                ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]}" for eid in range(len(doc_emos[id]))])

            ## ---set2--- 情感子句：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])
            ## ---set3--- 情感子句：ci：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'only_ecpe':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'ecpe_wo_label':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"{emotion}, {'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'wo_all_label':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"{doc_format_sents[id][doc_emos[id][eid]-1]},{doc_emocat[id][eid]},{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"{doc_sents[id][doc_emos[id][eid]-1]},{doc_emocat[id][eid]},{doc_emokeyword[id][eid]}" for eid in range(len(doc_emos[id]))])
            ## ---set2--- 情感子句：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])
            ## ---set3--- 情感子句：ci：XXX,情感类别：XXX
            # ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid]-1]},情感类别:{doc_emocat[id][eid]}" for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"{emotion}, {'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'wo_clause_types':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"{doc_format_sents[id][doc_emos[id][eid] - 1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"{doc_sents[id][doc_emos[id][eid] - 1]},情感类别:{doc_emocat[id][eid]},情感词:{doc_emokeyword[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])
            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"{emotion}, {'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'wo_emotion_types':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid] - 1]},情感词:{doc_emokeyword[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid] - 1]},情感词:{doc_emokeyword[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'wo_keywords':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid] - 1]},情感类别:{doc_emocat[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid] - 1]},情感类别:{doc_emocat[id][eid]}"
                                              for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)
    elif paradigm == 'wo_emotype_and_keywords':
        for id in range(len(doc_ids)):
            # document emotion extraction sub-task
            if formatted_clauseid == True:
                input_text = ','.join(doc_format_sents[id])
            else:
                input_text = ','.join(doc_sents[id])
            if formatted_clauseid == True:
                ## ---set1--- 情感子句：XXX,情感类别：XXX，情感词：XXX
                ee_target_text = ";".join([f"情感子句:{doc_format_sents[id][doc_emos[id][eid] - 1]}"
                                              for eid in range(len(doc_emos[id]))])
            else:
                ee_target_text = ";".join([f"情感子句:{doc_sents[id][doc_emos[id][eid] - 1]}"
                                              for eid in range(len(doc_emos[id]))])

            data.append({
                "task_type": "情感子句提取",
                "input_text": input_text,
                "target_text": ee_target_text
            })

            # ECPE sub-task
            emotion_cause_map = {}         # 创建一个字典，用于存储每个情感子句对应的所有原因子句
            for eid, cid in doc_ecps[id]:
                if formatted_clauseid == True:
                    emotion_sentence = doc_format_sents[id][eid - 1]
                    cause_sentence = doc_format_sents[id][cid - 1]
                else:
                    emotion_sentence = doc_sents[id][eid - 1]
                    cause_sentence = doc_sents[id][cid - 1]
                # 如果情感子句已经存在，则将新的原因子句添加到列表中
                if emotion_sentence in emotion_cause_map:
                    emotion_cause_map[emotion_sentence].append(cause_sentence)
                else:
                    emotion_cause_map[emotion_sentence] = [cause_sentence]
            # 将情感子句和对应的原因子句合并为预期的目标输出格式
            ecpe_target_text = ";".join(
                [f"情感子句:{emotion}, 原因子句:{'，'.join(causes)}" for emotion, causes in emotion_cause_map.items()]
            )
            data.append({
                "task_type": "情感原因对提取",
                "input_text": input_text,
                "target_text": ecpe_target_text
            })
        # print('data:', data)

    return data


def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} under {2} | {3:.4f} | ".format(
        args.task, args.dataset, args.paradigm, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        task_name = f'raw_scores_ecpe_{gstep}'
        for name in metric_names:
            name_step = f'{name}'
            results_str += f"{name:<8}: {dev_results[task_name][name_step]:.4f} / {test_results[task_name][name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)


class SINADataset(Dataset):
    def __init__(self, tokenizer, fold_id, data_type, task, paradigm, data_dir, clauseid, max_len=512):
        self.data_type = data_type
        self.task = task
        self.paradigm = paradigm
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.clauseid = clauseid
        if self.data_type == 'train':
            self.data_path = join('data/' + self.task + '/' + data_dir, TRAIN_FILE % fold_id)
        elif self.data_type == 'test':
            self.data_path = join('data/' + self.task + '/' + data_dir, TEST_FILE % fold_id)
        elif self.data_type == 'dev':
            self.data_path = join('data/' + self.task + '/' + data_dir, DEV_FILE % fold_id)
        elif self.data_type == 'val':
            self.data_path = join('data/' + self.task + '/' + data_dir, VALID_FILE % fold_id)
        self._build_examples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input_text"]
        target_text = item["target_text"]
        task_type = item["task_type"]
        # 拼接任务标识与输入文本
        input_encodings = self.tokenizer(
            # f"{task_type}: {input_text}",
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # 返回编码后的输入和目标
        return {
            "source_ids": input_encodings["input_ids"].squeeze(),
            "source_mask": input_encodings["attention_mask"].squeeze(),
            "target_ids": target_encodings["input_ids"].squeeze(),
            "target_mask": target_encodings["attention_mask"].squeeze(),
            "task_type": task_type
        }

    def _build_examples(self):
        data = get_sina_transformed_io(self.data_path, self.paradigm, self.task, self.clauseid)
        self.data = data


class NTCIRDataset(Dataset):
    def __init__(self, tokenizer, fold_id, data_type, task, paradigm, data_dir, clauseid, max_len=512):
        self.data_type = data_type
        self.task = task
        self.paradigm = paradigm
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.clauseid = clauseid  #
        # self.inputs = []
        # self.targets = []
        if self.data_type == 'train':
            self.data_path = join('data/' + self.task + '/' + data_dir, TRAIN_FILE % fold_id)
        elif self.data_type == 'val':
            self.data_path = join('data/' + self.task + '/' + data_dir, VALID_FILE % fold_id)
        elif self.data_type == 'test':
            self.data_path = join('data/' + self.task + '/' + data_dir, TEST_FILE % fold_id)
        self._build_examples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input_text"]
        target_text = item["target_text"]
        task_type = item["task_type"]
        # 拼接任务标识与输入文本
        input_encodings = self.tokenizer(
            # f"{task_type}: {input_text}",
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # 返回编码后的输入和目标
        return {
            "source_ids": input_encodings["input_ids"].squeeze(),
            "source_mask": input_encodings["attention_mask"].squeeze(),
            "target_ids": target_encodings["input_ids"].squeeze(),
            "target_mask": target_encodings["attention_mask"].squeeze(),
            "task_type": task_type
        }

    def _build_examples(self):
        data = get_ntcir_transformed_io(self.data_path, self.paradigm, self.task, self.clauseid)
        self.data = data

dev_data_path = join('data/' + 'ecpe' + '/' + 'eca_eng', 'fold1_val.txt')
_, _, dev_doc_sents, dev_doc_format_sents, _, _, _, _, _ = (read_ntcir_line_examples_from_file(dev_data_path))
print('dev_doc_sents:', dev_doc_sents)
print('dev_doc_format_sents:', dev_doc_format_sents)
