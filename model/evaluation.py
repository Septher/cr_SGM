from model.params import DEVICE_ORDER, K
import torch
import math

def evaluate(output_with_label, data_tag):
    result = dict()
    for index, device in enumerate(DEVICE_ORDER):
        outputs = torch.cat([output[index] for output, _ in output_with_label], dim=0)
        labels = torch.cat([label_dict[device] for _, label_dict in output_with_label], dim=0)
        result[device] = calculate(outputs, labels)
    show_result(result, data_tag)
    return result

def output_info(output_with_label):
    result = {}
    for index, device in enumerate(DEVICE_ORDER):
        outputs = torch.cat([output[index] for output, _ in output_with_label], dim=0)
        labels = torch.cat([label_dict[device] for _, label_dict in output_with_label], dim=0)
        indices = torch.argmax(outputs, dim=0)
        equals = indices == labels
        sample_cnt = outputs.size()[0]
        device_result = {}
        for i in range(sample_cnt):
            actual = labels[i].item()
            (true, total) = device_result.get(actual, (0, 0))
            device_result[actual] = (true + 1, total + 1) if equals[i].items() else (true, total + 1)
        for key in device_result.keys():
            (true, total) = device_result[key]
            device_result[key] = 1.0 * true / total
        result[device] = device_result
    return result

log2 = [1.0 / math.log2(k + 2) for k in range(K)]
def calculate(outputs, labels):
    sample_cnt = outputs.size()[0]
    tp, _DCG = [0] * K, [0] * K
    for i in range(sample_cnt):
        actual = labels[i].item()
        _, indices = torch.sort(outputs[i], descending=True)
        rk = torch.where(indices == actual, torch.ones_like(indices), torch.zeros_like(indices)).nonzero()[0].item()
        for k in range(K):
            if rk <= k:
                tp[k] += 1
                _DCG[k] += log2[rk]
    out = dict()
    for k in range(K):
        # recall@k = (number of top_k contains label) / sample_cnt
        out["recall@%d" % (k + 1)] = 1.0 * tp[k] / sample_cnt
        # precision@k = (number of top_k contains label) / (sample_cnt * k)
        out["precision@%d" % (k + 1)] = 1.0 * tp[k] / sample_cnt / (k + 1)
        # nDCG: iDCG = 1 because there are only one label for each sample
        out["nDCG@%d" % (k + 1)] = _DCG[k] / sample_cnt
    return out

def show_result(result, data_tag):
    print("%s training result: " % data_tag)
    recall_ave = 0.0
    for device in DEVICE_ORDER:
        for criterion in ["recall"]:
            print("%s %s: %s" % (device, criterion, str(["%.4f" % result[device]["%s@%d" % (criterion, k + 1)] for k in range(K)])))
        recall_ave += sum([result[device]["recall@%d" % (k + 1)] for k in range(K)])
    print("recall ave = %.4f" % (recall_ave / 25))
