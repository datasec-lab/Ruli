import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.utils.data import DataLoader

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
            except:
                device = (
                    torch.device("cuda:0")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                data, target = get_x_y_from_data_dict(batch, device)
            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    shadow_train_corr = (
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
        torch.argmax(target_train_prob, axis=1) == target_train_labels
    ).int()
    target_test_corr = (
        torch.argmax(target_test_prob, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)

    target_train_entr = entropy(target_train_prob)
    target_test_entr = entropy(target_test_prob)

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    if target_train is not None:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test is not None:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr

    acc_corr = SVC_fit_predict(
        shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    acc_entr = SVC_fit_predict(
        shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    )
    acc_m_entr = SVC_fit_predict(
        shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    )
    acc_prob = SVC_fit_predict(
        shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    )
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    print(m)
    return m



def basic_mia(model, forget_data, retain_data, test_data):

    evaluation_results = {}
    retain_data_loader = DataLoader(retain_data, batch_size=128, shuffle=False)
    forget_data_loader = DataLoader(forget_data, batch_size=128, shuffle=False)
    num = len(test_data) // 2

    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    test_len = len(test_data)
    forget_len = len(forget_data)
    retain_len = len(retain_data)

    shadow_train = torch.utils.data.Subset(retain_data, list(range(test_len)))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=128, shuffle=False
    )

    result_Df = SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=test_loader,
        target_train=None,
        target_test=forget_data_loader,
        model=model,
    )

    test_len = len(test_data)
    retain_len = len(retain_data)
    num = test_len // 2

    shadow_train = torch.utils.data.Subset(retain_data, list(range(num)))
    target_train = torch.utils.data.Subset(
        retain_data, list(range(num, retain_len))
    )
    shadow_test = torch.utils.data.Subset(test_data, list(range(num)))
    target_test = torch.utils.data.Subset(
        test_data, list(range(num, test_len))
    )

    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=128, shuffle=False
    )
    shadow_test_loader = torch.utils.data.DataLoader(
        shadow_test, batch_size=128, shuffle=False
    )

    target_train_loader = torch.utils.data.DataLoader(
        target_train, batch_size=128, shuffle=False
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=128, shuffle=False
    )

    result_Dr = SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=shadow_test_loader,
        target_train=target_train_loader,
        target_test=target_test_loader,
        model=model,
    )

    evaluation_results["forget"]=result_Df
    evaluation_results["remain"]=result_Dr

    return evaluation_results


def mia_threshold(model, tr_loader, te_loader, threshold, device='cuda:0', n_classes=10):

    with torch.inference_mode():
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        model.eval()
        tp, fp = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        tn, fn = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)

        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True), \
                labels.to(device=device, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                tp[i] += n_member_pred
                fn[i] += len(preds) - n_member_pred

        # on test loader (non-members, i.e., negative class)
        for _, (inputs, labels) in enumerate(te_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True), \
                labels.to(device=device, non_blocking=True)
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                fp[i] += n_member_pred
                tn[i] += len(preds) - n_member_pred

        # class-wise bacc, tpr, fpr computations
        class_tpr, class_fpr = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        class_bacc = torch.zeros(n_classes, device=device)
        for i in range(n_classes):
            class_i_tpr, class_i_tnr = tp[i] / (tp[i] + fn[i]), tn[i] / (tn[i] + fp[i])
            class_tpr[i], class_fpr[i] = class_i_tpr, 1 - class_i_tnr
            class_bacc[i] = (class_i_tpr + class_i_tnr) / 2

        # dataset-wise bacc, tpr, fpr computations
        ds_tp, ds_fp = tp.sum(), fp.sum()
        ds_tn, ds_fn = tn.sum(), fn.sum()
        ds_tpr, ds_tnr = ds_tp / (ds_tp + ds_fn), ds_tn / (ds_tn + ds_fp)
        ds_bacc, ds_fpr = (ds_tpr + ds_tnr) / 2, 1 - ds_tnr

    return (ds_bacc, ds_tpr, ds_fpr), (class_bacc, class_tpr, class_fpr)