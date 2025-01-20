# -*- coding: utf-8 -*-
# This file contains the evaluation functions
from difflib import SequenceMatcher

clause_list = ['emotion clause', 'cause clause', '情感子句', '原因子句']
emotion_category_list = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear']

def extract_spans_multitask(task, seq, dataset):
    print('seq:', seq)
    if task in ['ecpe']:
        ecpe_extractions, ee_extractions, ce_extractions = [], [], []
        all_pt = [pt.strip() for pt in seq.split(';')]  # 去除分割后的多余空格
        # print('all_pt:', all_pt)
        for pt in all_pt:
            if dataset == 'eca_eng':
                e_c_sign = "emotion clause:"
                c_c_sign = "cause clause:"
            else:
                e_c_sign = "情感子句:"
                c_c_sign = "原因子句:"
            try:
                all_contence = [item.strip() for item in pt.split(',')]
                print('all_contence:', all_contence)
                e_cla_label, e_clause = all_contence[0].split(':', 1)
                # print('e_cla_label, e_clause:', e_cla_label, e_clause)
                if len(all_contence) != 1:
                    cau_contents = all_contence[1:]
                    # print('cau_contents:', cau_contents)
                    # 提取原因标签和内容
                    cau_label, first_cause = cau_contents[0].split(':', 1)
                    all_causes = [first_cause] + cau_contents[1:] if len(cau_contents) > 1 else [first_cause]

                    # 合并所有原因内容
                    all_causes[0] = c_c_sign + all_causes[0]
                else:
                    all_causes = c_c_sign + 'none'
            except ValueError:
                # 如果发生错误，跳过此项或执行特定的处理
                continue
            ecpe_extractions.append(all_contence)
            ee_extractions.append(e_c_sign+e_clause)
            ce_extractions.append(all_causes)
            # print('ecpe_extractions:', ecpe_extractions)
        return [ecpe_extractions, ee_extractions, ce_extractions]


def extract_spans_multitask_mee(task, seq):
    # print('seq:', seq)
    if task in ['ecpe']:
        ecpe_extractions, ee_extractions, ce_extractions = [], [], []
        all_pt = [pt.strip() for pt in seq.split(';')]  # 去除分割后的多余空格
        # print('all_pt:', all_pt)
        for pt in all_pt:
            try:
                all_contence = [item.strip() for item in pt.split(',')]
                # print('all_contence:', all_contence)
                e_cla_label, e_clause = all_contence[0].split(':', 1)
                # print('e_cla_label, e_clause:', e_cla_label, e_clause)
                if len(all_contence) != 1:
                    cau_contents = all_contence[1:]
                    # print('cau_contents:', cau_contents)
                    # 提取原因标签和内容
                    cau_label, first_cause = cau_contents[0].split(':', 1)
                    all_causes = [first_cause] + cau_contents[1:] if len(cau_contents) > 1 else [first_cause]

                    # 合并所有原因内容
                    all_causes[0] = '原因子句:' + all_causes[0]
                else:
                    all_causes = '原因子句:' + 'none'
            except ValueError:
                # 如果发生错误，跳过此项或执行特定的处理
                continue
            ecpe_extractions.append(all_contence)
            ee_extractions.append('情感子句:'+e_clause)
            ce_extractions.append(all_causes)
            # print('ecpe_extractions:', ecpe_extractions)
        return [ecpe_extractions, ee_extractions, ce_extractions]


def extract_spans_multitask_wo_label(task, seq):
    # print('seq:', seq)
    if task in ['ecpe']:
        ecpe_extractions, ee_extractions, ce_extractions = [], [], []
        all_pt = [pt.strip() for pt in seq.split(';')]  # 去除分割后的多余空格
        # print('all_pt:', all_pt)
        for pt in all_pt:
            try:
                all_contence = [item.strip() for item in pt.split(',')]
                # print('all_contence:', all_contence)
                # e_cla_label, e_clause = all_contence[0].split(':', 1)
                e_clause = all_contence[0]
                # print('e_cla_label, e_clause:', e_cla_label, e_clause)
                if len(all_contence) != 1:
                    cau_contents = all_contence[1:]
                    # print('cau_contents:', cau_contents)
                    # 提取原因标签和内容
                    # cau_label, first_cause = cau_contents[0].split(':', 1)
                    first_cause = cau_contents[0]
                    all_causes = [first_cause] + cau_contents[1:] if len(cau_contents) > 1 else [first_cause]
                    # 合并所有原因内容
                    # all_causes[0] = '原因子句:' + all_causes[0]
                else:
                    # all_causes = '原因子句:' + 'none'
                    all_causes = 'none'

            except ValueError:
                # 如果发生错误，跳过此项或执行特定的处理
                continue
            ecpe_extractions.append(all_contence)
            # ee_extractions.append('情感子句:'+e_clause)
            ee_extractions.append(e_clause)
            ce_extractions.append(all_causes)
            # print('ecpe_extractions:', ecpe_extractions)
        return [ecpe_extractions, ee_extractions, ce_extractions]


def recover_terms_with_difflib_sequencematcher(p_clauses, o_sents, threshold=0.5):
    """
    找到与预测子句最相似且超过相似度阈值的文档句子
    """
    best_match = p_clauses  # 默认设置为预测子句自身
    highest_similarity = 0
    # print('o_sents:', o_sents)

    for sentence in o_sents:
        similarity = SequenceMatcher(None, p_clauses, sentence).ratio()
        # print('sentence, similarity:', sentence, similarity)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = sentence

    # 如果找到的最佳匹配超过阈值，则返回，否则返回原句
    return best_match if highest_similarity >= threshold else p_clauses


def fix_multitask_preds_ecpe(all_pairs, sents, dataset):
    # print('all_pairs:', all_pairs)
    ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs = [], [], []

    for i, pairs in enumerate(all_pairs):
        new_pairs, new_ee, new_ce = [], [], []

        # 如果没有情感原因对
        if not pairs[0]:
            ecpe_all_new_pairs.append(pairs[0])
            ee_all_new_pairs.append(pairs[0])
            ce_all_new_pairs.append(pairs[0])
        else:
            # 遍历所有情感-原因对
            for pair in pairs[0]:
                # print('pair:', pair)
                pair_ee, pair_ce = [], []
                # 处理情感子句
                ecla_label, e_clause = pair[0].split(':', 1)
                if ecla_label not in clause_list:
                    new_clause_emo_cat = recover_terms_with_difflib_sequencematcher(ecla_label, clause_list)
                else:
                    new_clause_emo_cat = ecla_label
                if e_clause not in sents[i]:
                    new_emo_clause = recover_terms_with_difflib_sequencematcher(e_clause, sents[i])
                else:
                    new_emo_clause = e_clause
                new_emo_part = new_clause_emo_cat + ':' + new_emo_clause
                pair_ee.append(new_emo_part)

                # 处理原因子句（可能有多个原因子句）
                if len(pair) != 1:
                    cau_contents = pair[1:]
                    cau_label, first_cause = cau_contents[0].split(':', 1)
                    all_causes = [first_cause] + cau_contents[1:] if len(cau_contents) > 1 else [first_cause]
                    if cau_label not in clause_list:
                        new_clause_cau_cat = recover_terms_with_difflib_sequencematcher(cau_label, clause_list)
                    else:
                        new_clause_cau_cat = cau_label
                    cau_clauses = []
                    for clause_i in all_causes:
                        if clause_i not in sents[i]:
                            new_cau_clause = recover_terms_with_difflib_sequencematcher(clause_i, sents[i])
                        else:
                            new_cau_clause = clause_i
                        cau_clauses.append(new_cau_clause)
                    pair_ce.append(cau_clauses)
                    cau_clauses[0] = new_clause_cau_cat + ':' + cau_clauses[0]
                else:
                    if dataset == 'eca_eng':
                        cau_clauses = ['cause clause:' + 'None']
                    else:
                        cau_clauses = ['原因子句:' + 'None']
                    pair_ce.append(cau_clauses)


                # 添加情感-原因对到新列表中
                new_pairs.append([new_emo_part] + cau_clauses)
                new_ee.extend(pair_ee)
                new_ce.append(pair_ce if len(pair_ce) > 1 else pair_ce[0])

            # 添加处理后的结果到输出列表
            ecpe_all_new_pairs.append(new_pairs)
            ee_all_new_pairs.append(new_ee)
            ce_all_new_pairs.append(new_ce)
    return [ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs]


def fix_multitask_preds_ecpe_wo_label(all_pairs, sents):
    # print('all_pairs:', all_pairs)
    ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs = [], [], []

    for i, pairs in enumerate(all_pairs):
        new_pairs, new_ee, new_ce = [], [], []

        # 如果没有情感原因对
        if not pairs[0]:
            ecpe_all_new_pairs.append(pairs[0])
            ee_all_new_pairs.append(pairs[0])
            ce_all_new_pairs.append(pairs[0])
        else:
            # 遍历所有情感-原因对
            for pair in pairs[0]:
                # print('pair:', pair)
                pair_ee, pair_ce = [], []
                # 处理情感子句
                e_clause = pair[0]
                if e_clause not in sents[i]:
                    new_emo_clause = recover_terms_with_difflib_sequencematcher(e_clause, sents[i])
                else:
                    new_emo_clause = e_clause
                new_emo_part = new_emo_clause
                pair_ee.append(new_emo_part)

                # 处理原因子句（可能有多个原因子句）
                if len(pair) != 1:
                    cau_contents = pair[1:]
                    first_cause = cau_contents[0]
                    all_causes = [first_cause] + cau_contents[1:] if len(cau_contents) > 1 else [first_cause]
                    cau_clauses = []
                    for clause_i in all_causes:
                        if clause_i not in sents[i]:
                            new_cau_clause = recover_terms_with_difflib_sequencematcher(clause_i, sents[i])
                        else:
                            new_cau_clause = clause_i
                        cau_clauses.append(new_cau_clause)
                    pair_ce.append(cau_clauses)
                else:
                    cau_clauses = ['None']
                    pair_ce.append(cau_clauses)


                # 添加情感-原因对到新列表中
                new_pairs.append([new_emo_part] + cau_clauses)
                new_ee.extend(pair_ee)
                new_ce.append(pair_ce if len(pair_ce) > 1 else pair_ce[0])

            # 添加处理后的结果到输出列表
            ecpe_all_new_pairs.append(new_pairs)
            ee_all_new_pairs.append(new_ee)
            ce_all_new_pairs.append(new_ce)
    return [ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs]


def fix_pred_with_editdistance(all_predictions, sents, task, io_format, dataset):
    if task == 'ecpe':
        if io_format in ['multi_task', 'multi_task_mee', 'only_ecpe', 'wo_emotion_types',
                         'wo_keywords', 'wo_emotype_and_keywords']:
            fixed_preds = fix_multitask_preds_ecpe(all_predictions, sents, dataset)
            return fixed_preds
        elif io_format in ['wo_all_label', 'wo_clause_types']:
            fixed_preds = fix_multitask_preds_ecpe_wo_label(all_predictions, sents)
            return fixed_preds
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions
        return fixed_preds


def fix_emo_pred_with_editdistance(all_predictions, sents, task, io_format, dataset):
    if task == 'ecpe':
        if io_format == 'multi_task':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        if dataset == 'eca_eng':
                            e_c_sign = "emotion clause:"
                            e_cat_sign = "emotion type:"
                            e_kw_sign = "clue words:"
                        else:
                            e_c_sign = "情感子句:"
                            e_cat_sign = "情感类别:"
                            e_kw_sign = "情感词:"
                        if e_c_sign in all_emo_cla:
                            emo_clause_start = all_emo_cla.find(e_c_sign) + len(e_c_sign)
                            emo_clause_end = all_emo_cla.find(","+e_cat_sign, emo_clause_start)
                            emo_clause = all_emo_cla[emo_clause_start:emo_clause_end if emo_clause_end != -1 else None]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = e_c_sign
                            emo_cla_part = new_emo_clause_cat + new_emo_clause
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = e_c_sign
                            emo_cla_part = new_emo_clause_cat + new_emo_clause
                        if e_cat_sign in all_emo_cla:
                            emo_clause_cat_start = all_emo_cla.find(e_cat_sign) + len(e_cat_sign)
                            emo_clause_cat_end = all_emo_cla.find(","+e_kw_sign, emo_clause_cat_start)
                            cat = all_emo_cla[emo_clause_cat_start:emo_clause_cat_end if emo_clause_cat_end != -1 else None]
                            new_emo_cat = e_cat_sign
                            if cat not in emotion_category_list:
                                new_cat = recover_terms_with_difflib_sequencematcher(cat, emotion_category_list)
                            else:
                                new_cat = cat
                            emo_cat_part = new_emo_cat + new_cat
                        else:
                            new_emo_cat = e_cat_sign
                            emo_cat_part = new_emo_cat + 'None'
                        # 情感关键词
                        if e_kw_sign in all_emo_cla:
                            emo_keyword_start = all_emo_cla.find(e_kw_sign) + len(e_kw_sign)
                            emokeyword = all_emo_cla[emo_keyword_start:]
                            new_emo_kword = e_kw_sign
                            emo_keyword_part = new_emo_kword + emokeyword
                        else:
                            new_emo_kword = e_kw_sign
                            emo_keyword_part = new_emo_kword + 'none'
                        new_ee.append(emo_cla_part + ',' + emo_cat_part + ',' + emo_keyword_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'wo_all_label':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        if len(all_emo_cla) == 3:
                            emo_clause, cat, emokeyword = all_emo_cla.split(',')
                            # 情感子句
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            emo_cla_part = new_emo_clause
                            # cat
                            if cat not in emotion_category_list:
                                new_cat = recover_terms_with_difflib_sequencematcher(cat, emotion_category_list)
                            else:
                                new_cat = cat
                            emo_cat_part = new_cat
                            # 情感关键词
                            emo_keyword_part = emokeyword
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            # 情感子句
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            emo_cla_part = new_emo_clause
                            emo_cat_part = "none"
                            # 情感关键词
                            emo_keyword_part = "none"
                        new_ee.append(emo_cla_part + ',' + emo_cat_part + ',' + emo_keyword_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'multi_task_mee':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        if "情感子句:" in all_emo_cla:
                            emo_clause_start = all_emo_cla.find("情感子句:") + len("情感子句:")
                            emo_clause_end = all_emo_cla.find(",情感类别:", emo_clause_start)
                            emo_clause = all_emo_cla[emo_clause_start:emo_clause_end if emo_clause_end != -1 else None]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause

                        # 情感类别：cat
                        if "情感类别:" in all_emo_cla:
                            emo_clause_cat_start = all_emo_cla.find("情感类别:") + len("情感类别:")
                            emo_clause_cat_end = all_emo_cla.find(",情感词:", emo_clause_cat_start)
                            cat = all_emo_cla[emo_clause_cat_start:emo_clause_cat_end if emo_clause_cat_end != -1 else None]
                            new_emo_cat = '情感类别'
                            if cat not in emotion_category_list:
                                new_cat = recover_terms_with_difflib_sequencematcher(cat, emotion_category_list)
                            else:
                                new_cat = cat
                            emo_cat_part = new_emo_cat + ':' + new_cat
                        else:
                            new_emo_cat = '情感类别'
                            emo_cat_part = new_emo_cat + ':' + 'None'
                        # 情感关键词
                        if "情感词:" in all_emo_cla:
                            emo_keyword_start = all_emo_cla.find("情感词:") + len("情感词:")
                            emokeyword = all_emo_cla[emo_keyword_start:]
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + emokeyword
                        else:
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + 'none'
                        new_ee.append(emo_cla_part + ',' + emo_cat_part + ',' + emo_keyword_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'wo_clause_types':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        emo_clause = all_emo_cla.split(',')[0]
                        if emo_clause not in sents[i]:
                            new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                        else:
                            new_emo_clause = emo_clause
                        emo_cla_part = new_emo_clause

                        # 情感类别：cat
                        if "情感类别:" in all_emo_cla:
                            emo_clause_cat_start = all_emo_cla.find("情感类别:") + len("情感类别:")
                            emo_clause_cat_end = all_emo_cla.find(",情感词:", emo_clause_cat_start)
                            cat = all_emo_cla[emo_clause_cat_start:emo_clause_cat_end if emo_clause_cat_end != -1 else None]
                            new_emo_cat = '情感类别'
                            if cat not in emotion_category_list:
                                new_cat = recover_terms_with_difflib_sequencematcher(cat, emotion_category_list)
                            else:
                                new_cat = cat
                            emo_cat_part = new_emo_cat + ':' + new_cat
                        else:
                            new_emo_cat = '情感类别'
                            emo_cat_part = new_emo_cat + ':' + 'None'
                        # 情感关键词
                        if "情感词:" in all_emo_cla:
                            emo_keyword_start = all_emo_cla.find("情感词:") + len("情感词:")
                            emokeyword = all_emo_cla[emo_keyword_start:]
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + emokeyword
                        else:
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + 'none'
                        new_ee.append(emo_cla_part + ',' + emo_cat_part + ',' + emo_keyword_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'wo_emotion_types':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        if "情感子句:" in all_emo_cla:
                            emo_clause_start = all_emo_cla.find("情感子句:") + len("情感子句:")
                            emo_clause_end = all_emo_cla.find(",情感类别:", emo_clause_start)
                            emo_clause = all_emo_cla[emo_clause_start:emo_clause_end if emo_clause_end != -1 else None]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        # 情感关键词
                        if "情感词:" in all_emo_cla:
                            emo_keyword_start = all_emo_cla.find("情感词:") + len("情感词:")
                            emokeyword = all_emo_cla[emo_keyword_start:]
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + emokeyword
                        else:
                            new_emo_kword = '情感词'
                            emo_keyword_part = new_emo_kword + ':' + 'none'
                        new_ee.append(emo_cla_part + ',' + emo_keyword_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'wo_keywords':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        if "情感子句:" in all_emo_cla:
                            emo_clause_start = all_emo_cla.find("情感子句:") + len("情感子句:")
                            emo_clause_end = all_emo_cla.find(",情感类别:", emo_clause_start)
                            emo_clause = all_emo_cla[emo_clause_start:emo_clause_end if emo_clause_end != -1 else None]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause

                        # 情感类别：cat
                        if "情感类别:" in all_emo_cla:
                            emo_clause_cat_start = all_emo_cla.find("情感类别:") + len("情感类别:")
                            emo_clause_cat_end = all_emo_cla.find(",情感词:", emo_clause_cat_start)
                            cat = all_emo_cla[emo_clause_cat_start:emo_clause_cat_end if emo_clause_cat_end != -1 else None]
                            new_emo_cat = '情感类别'
                            if cat not in emotion_category_list:
                                new_cat = recover_terms_with_difflib_sequencematcher(cat, emotion_category_list)
                            else:
                                new_cat = cat
                            emo_cat_part = new_emo_cat + ':' + new_cat
                        else:
                            new_emo_cat = '情感类别'
                            emo_cat_part = new_emo_cat + ':' + 'None'
                        # # 情感关键词
                        # emokeyword = all_emo_cla.split(',')[-1]
                        # emo_keyword_part = emokeyword
                        new_ee.append(emo_cla_part + ',' + emo_cat_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        elif io_format == 'wo_emotype_and_keywords':
            ee_all_new = []
            for i, all_emocla in enumerate(all_predictions):
                # print('all_emocla:', all_emocla)
                all_emo_cla_l = all_emocla.split(';')
                # 如果没有情感子句
                if not all_emo_cla_l:
                    ee_all_new.append(all_emo_cla_l)
                else:
                    new_ee = []
                    for all_emo_cla in all_emo_cla_l:
                        # print('all_emo_cla:', all_emo_cla)
                        # 情感子句类别：情感子句
                        if "情感子句:" in all_emo_cla:
                            emo_clause_start = all_emo_cla.find("情感子句:") + len("情感子句:")
                            emo_clause_end = all_emo_cla.find(",情感类别:", emo_clause_start)
                            emo_clause = all_emo_cla[emo_clause_start:emo_clause_end if emo_clause_end != -1 else None]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        else:
                            emo_clause = all_emo_cla.split(',')[0]
                            if emo_clause not in sents[i]:
                                new_emo_clause = recover_terms_with_difflib_sequencematcher(emo_clause, sents[i])
                            else:
                                new_emo_clause = emo_clause
                            new_emo_clause_cat = '情感子句'
                            emo_cla_part = new_emo_clause_cat + ':' + new_emo_clause
                        new_ee.append(emo_cla_part)
                    ee_all_new.append(';'.join(new_ee) if len(new_ee) > 1 else new_ee[0])
        return ee_all_new
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions
        return fixed_preds


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task, dataset):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if task in ['ecpe']:
            if io_format in ['multi_task', 'only_ecpe', 'wo_emotion_types', 'wo_keywords', 'wo_emotype_and_keywords']:
                gold_list = extract_spans_multitask(task, gold_seqs[i], dataset)
                pred_list = extract_spans_multitask(task, pred_seqs[i], dataset)
            elif io_format in ['multi_task_mee', '']:
                gold_list = extract_spans_multitask_mee(task, gold_seqs[i])
                pred_list = extract_spans_multitask_mee(task, pred_seqs[i])
            elif io_format in ['wo_all_label', 'wo_clause_types']:
                gold_list = extract_spans_multitask_wo_label(task, gold_seqs[i])
                pred_list = extract_spans_multitask_wo_label(task, pred_seqs[i])
            all_labels.append(gold_list)
            all_predictions.append(pred_list)
    if task in ['ecpe']:
        print("\nResults of raw output")
        ecpe_all_p, ee_all_p, ce_all_p = [], [], []
        ecpe_all_l, ee_all_l, ce_all_l = [], [], []
        # print('all_labels:', all_labels)
        for i, tri_re in enumerate(all_predictions):
            ecpe_all_p.append(tri_re[0])
            ee_all_p.append(tri_re[1])
            ce_all_p.append(tri_re[2])
        for i, tri_label in enumerate(all_labels):
            ecpe_all_l.append(tri_label[0])
            ee_all_l.append(tri_label[1])
            ce_all_l.append(tri_label[2])
        raw_scores_ecpe = compute_f1_scores(ecpe_all_p, ecpe_all_l)
        raw_scores_ee = compute_f1_scores(ee_all_p, ee_all_l)
        raw_scores_ce = compute_f1_scores(ce_all_p, ce_all_l)
        print('raw_scores_ecpe：', raw_scores_ecpe)
        print('raw_scores_ee:', raw_scores_ee)
        print('raw_scores_ce:', raw_scores_ce)
        # print(raw_scores)
        print('all_labels:', all_labels)

        # fix the issues due to generation
        all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task, io_format, dataset)
        print('all_fix:', all_predictions_fixed[0])
        print('all_labels:', ecpe_all_l)

        print("\nResults of fixed output")
        fixed_scores_ecpe = compute_f1_scores(all_predictions_fixed[0], ecpe_all_l)
        print('fixed_scores_ecpe:', fixed_scores_ecpe)
        fixed_scores_ee = compute_f1_scores(all_predictions_fixed[1], ee_all_l)
        print('fixed_scores_ee:', fixed_scores_ee)
        fixed_scores_ce = compute_f1_scores(all_predictions_fixed[2], ce_all_l)
        print('fixed_scores_ce:', fixed_scores_ce)

        log_file_path = f"results_log/{task}-{io_format}.txt"
        with open(log_file_path, "a+", encoding='utf-8') as f:
            f.write('----ecpe---all_labels:\n' + str(all_labels) + '\n')
            f.write('----ecpe---all_prediction:\n' + str(all_predictions) + '\n'
                    + '----ecpe----fixed_all_prediction:\n' + str(all_predictions_fixed) + '\n')

        return raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce, \
               all_labels, all_predictions, all_predictions_fixed


def compute_scores_emotask(pred_seqs, gold_seqs, sents, io_format, task, dataset):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    all_labels, all_predictions = gold_seqs, pred_seqs


    print("\nEE Results of raw output")
    raw_scores_ee = compute_f1_scores(all_predictions, all_labels)
    print('raw_scores_ee:', raw_scores_ee)

    # print('all_labels:', all_labels)
    # fix the issues due to generation
    all_predictions_fixed = fix_emo_pred_with_editdistance(all_predictions, sents, task, io_format, dataset)
    # print('all_fix:', all_predictions_fixed)

    print("\nEE Results of fixed output")
    fixed_scores_ee = compute_f1_scores(all_predictions_fixed, all_labels)

    log_file_path = f"results_log/{task}-{io_format}.txt"
    with open(log_file_path, "a+", encoding='utf-8') as f:
        f.write('----ee----all label:\n' + str(all_labels) + '\n')
        f.write('----ee----all prediction:\n' + str(all_predictions) + '\n'
                + '----ee----fix all prediction:\n' + str(all_predictions_fixed) + '\n')

    return raw_scores_ee, fixed_scores_ee, all_labels, all_predictions, all_predictions_fixed


pred_seqs = [
    "[emotion clause:c1:He told me he was deeply angry about the fact that he had been given up., cause clause:c1:He told me he was deeply angry about the fact that he had been given up.]",
    "[emotion clause:c3:Jobs was so impressed and grateful that he offered Wayne a 10% stake in the new partnership, cause clause:c2:and this required him to commit his designs to the partnership]",
    "[emotion clause:c2:he railed, cause clause:c1:We dont have a web to spare]",
    "[emotion clause:c2:and so he shocked Wozniak by paying little, cause clause:c1:He wanted to secured a rehearsal right at the front of the hall as a unexpected way to release the Apple II]",
    "[emotion clause:c7:Jobs threw a quickerum, cause clause:c1:Scott assigned 1 to Wozniak and 2 to Jobs]"
]

gold_seqs = [
    "[emotion clause:c1:He told me he was deeply angry about the fact that he had been given up., cause clause:c1:He told me he was deeply angry about the fact that he had been given up.]",
    "[emotion clause:c3:Jobs was so impressed and grateful that he offered Wayne a 10% stake in the new partnership, cause clause:c2:and this required him to commit his designs to the partnership]",
    "[emotion clause:c2:he railed, cause clause:c1:We dont have a web to spare]",
    "[emotion clause:c2:and so he shocked Wozniak by paying little, cause clause:c1:He wanted to secured a rehearsal right at the front of the hall as a unexpected way to release the Apple II]",
    "[emotion clause:c7:Jobs threw a quickerum, cause clause:c1:Scott assigned 1 to Wozniak and 2 to Jobs]"
]

sents = ["c1:He told me he was deeply angry about the fact that he had been given up.",
         "c1:His argument was that a great engineer would be remembered only if he teamed with a great marketer, c2:and this required him to commit his designs to the partnership, c3:Jobs was so impressed and grateful that he offered Wayne a 10% stake in the new partnership, c4:turning him into a tie-breaker if Jobs and Wozniak disagreed over an issue.",
         "c1:We dont have a chip to spare, c2:he railed, c3:correctly.",
         "c1:He wanted to secure a location right at the front of the hall as a dramatic way to launch the Apple II, c2:and so he shocked Wozniak by paying $5, c3:000 in advance.",
         "c1:Scott assigned  1 to Wozniak and  2 to Jobs, c2:Not surprisingly, c3:Jobs demanded to be  1, c4:I would n't let him have it, c5:because that would stoke his ego even more, c6:said Scott, c7:Jobs threw a tantrum, c8:even cried.",
        ]

in_format = 'multi_task'
task = 'ecpe'
dataset = 'eca_eng'
raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(pred_seqs, gold_seqs, sents, in_format, task, dataset)

