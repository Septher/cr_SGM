from data.dataset_helper import devices_order
import torch
import math

K = 5

def evaluate(output_with_label, data_tag):
    result = dict()
    for index, device in enumerate(devices_order):
        # output_zip_label = [(output[index], label_dict[device]) for output, label_dict in output_with_label]
        outputs = torch.cat([output[index] for output, _ in output_with_label], dim=0)
        labels = torch.cat([label_dict[device] for _, label_dict in output_with_label], dim=0)
        result[device] = calculate(outputs, labels)
    show_result(result, data_tag)
    return result

log2 = [1.0 / math.log2(k + 2) for k in range(K)]
def calculate(outputs, labels):
    sample_cnt = outputs.size()[0]
    tp, _DCG = [0] * K, [0] * K
    for i in range(sample_cnt):
        actual = labels[i].item()
        _, indices = torch.sort(outputs[i], descending=True)
        pos = torch.where(indices == actual, indices + 1, torch.zeros_like(indices)).max().item() - 1
        for k in range(K):
            if pos <= k:
                tp[k] += 1
                _DCG[k] += log2[pos]
    out = dict()
    for k in range(K):
        # recall@k = (number of top_k contains label) / sample_cnt
        out["recall@%d" % (k + 1)] = 1.0 * tp[k] / sample_cnt
        # precision@k = (number of top_k contains label) / (sample_cnt * k)
        out["precision@%d" % (k + 1)] = 1.0 * tp[k] / sample_cnt / (k + 1)
        # nDCG: iDCG = 1 because there are only one label for each sample
        out["nDCG@%d" % (k + 1)] = _DCG[k]
    return out

def show_result(result, data_tag):
    print("%s training result: " % data_tag)
    for device in devices_order:
        for criterion in ["recall"]:
            print("%s %s: %s" % (device, criterion, str(["%.4f" % result[device]["%s@%d" % (criterion, k + 1)] for k in range(K)])))
