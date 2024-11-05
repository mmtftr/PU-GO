#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES, EXP_CODES


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def test(data_root, ont, model, run, combine, alpha, tex_output, wandb_logger):
    """Main evaluation function that computes performance metrics for GO term prediction.

    Args:
        data_root: Root directory containing data files
        ont: Ontology type (mf, bp, cc)
        model: Name of model being evaluated
        run: Run number
        combine: Whether to combine predictions with Diamond predictions
        alpha: Weight for combining Diamond and model predictions
        tex_output: Whether to output results in LaTeX format
        wandb_logger: Logger for tracking metrics with Weights & Biases
    """
    # Load data files
    train_data_file = f"{data_root}/{ont}/train_data.pkl"
    valid_data_file = f"{data_root}/{ont}/valid_data.pkl"
    test_data_file = f"{data_root}/{ont}/predictions_{model}_{run}.pkl"
    if combine:
        diam_data_file = f"{data_root}/{ont}/time_data_diam.pkl"
        diam_df = pd.read_pickle(diam_data_file)

    # Load GO terms and create mapping
    terms_file = f"{data_root}/{ont}/terms.pkl"
    go_rels = Ontology(f"{data_root}/go-basic.obo", with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df["gos"].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    # Load train/valid/test data
    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)

    # Calculate information content from annotations
    annotations = train_df["prop_annotations"].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df["prop_annotations"].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Get IC values for each term
    ics = {}
    for i, term in enumerate(terms):
        ics[term] = go_rels.get_ic(term)

    # Combine model predictions with Diamond predictions if specified
    eval_preds = []
    for i, row in enumerate(test_df.itertuples()):
        if combine:
            diam_preds = np.zeros((len(terms),), dtype=np.float32)
            for go_id, score in diam_df.iloc[i]["diam_preds"].items():
                if go_id in terms_dict:
                    diam_preds[terms_dict[go_id]] = score

            preds = diam_preds * alpha + row.preds * (1 - alpha)
        else:
            preds = row.preds
        eval_preds.append(preds)

    # Create label matrix and prediction matrix
    labels = np.zeros((len(test_df), len(terms)), dtype=np.float32)
    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))

    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1

    # Calculate average AUC across terms
    total_n = 0
    total_sum = 0
    for go_id, i in terms_dict.items():
        pos_n = np.sum(labels[:, i])
        if pos_n > 0 and pos_n < len(test_df):
            total_n += 1
            roc_auc = compute_roc(labels[:, i], eval_preds[:, i])
            total_sum += roc_auc

    avg_auc = total_sum / total_n

    # Calculate Fmax and other metrics across different thresholds
    print("Computing Fmax")
    fmax = 0.0
    tmax = 0.0
    wfmax = 0.0
    wtmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df["prop_annotations"].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    spec_labels = test_df["exp_annotations"].values
    spec_labels = list(
        map(lambda x: set(filter(lambda y: y in go_set, x)), spec_labels)
    )
    fmax_spec_match = 0

    # Try different thresholds to find optimal Fmax
    for t in range(0, 101):
        threshold = t / 100.0
        preds = [set() for _ in range(len(test_df))]
        for i in range(len(test_df)):
            annots = set()
            above_threshold = np.argwhere(eval_preds[i] >= threshold).flatten()
            for j in above_threshold:
                annots.add(terms[j])

            if t == 0:
                preds[i] = annots
                continue
            preds[i] = annots

        # Filter predictions to relevant namespace
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(
            go_rels, labels, preds
        )

        # Calculate matches with experimental annotations
        spec_match = 0
        for i, row in enumerate(test_df.itertuples()):
            spec_match += len(spec_labels[i].intersection(preds[i]))

        precisions.append(prec)
        recalls.append(rec)

        # Track best scores and thresholds
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            avgic = avg_ic
            fmax_spec_match = spec_match
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s

    # Append _diam to model name if using combined predictions
    if combine:
        model += "_diam"

    # Print results
    print(model, ont)
    print(
        f"Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}"
    )
    print(f"WFmax: {wfmax:0.3f}, threshold: {wtmax}")
    print(f"AUC: {avg_auc:0.3f}")

    # Calculate AUPR
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f"AUPR: {aupr:0.3f}")
    print(f"AVGIC: {avgic:0.3f}")

    # Log metrics to wandb
    if combine:
        wandb_logger.log(
            {
                "fmax_diam": fmax,
                "smin_diam": smin,
                "aupr_diam": aupr,
                "avg_auc_diam": avg_auc,
                "wfmax_diam": wfmax,
                "avgic_diam": avgic,
                "threshold_diam": tmax,
                "w_threshold_diam": wtmax,
                "spec_diam": fmax_spec_match,
                "combine_diam": combine,
            }
        )
    else:
        wandb_logger.log(
            {
                "fmax": fmax,
                "smin": smin,
                "aupr": aupr,
                "avg_auc": avg_auc,
                "wfmax": wfmax,
                "avgic": avgic,
                "threshold": tmax,
                "w_threshold": wtmax,
                "spec": fmax_spec_match,
                "combine": combine,
            }
        )

    # Output results in LaTeX format if requested
    if tex_output:
        tex = "& "
        tex += f"{fmax:0.3f} & {smin:0.3f} & {aupr:0.3f} & {avg_auc:0.3f} \\\\"
        print(tex)


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            if tpic + fpic > 0:
                wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf


if __name__ == "__main__":
    main()
