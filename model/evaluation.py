from data.dataset_helper import devices_order
import torch

def evaluate(output_with_label, data_tag):
    result = dict()
    for index, device in enumerate(devices_order):
        output_zip_label = [(output[index], task_dict[device]) for output, task_dict in output_with_label]

        device_dict = dict()
        for k in range(1, 6):
            # device_dict["precision@%d" % k] = precision(output_zip_label, k, data_tag)
            device_dict["recall@%d" % k] = recall(output_zip_label, k, data_tag)
            # device_dict["nDCG@%d" % k] = nDCG(output_zip_label, k, data_tag)
        result[device] = device_dict
    return result

def recall(output_zip_label, k, data_tag):
    sample_cnt = 0
    tp, fp, fn = 0, 0, 0
    for output, label in output_zip_label:
        batch_size = output.shape[0]
        sample_cnt += batch_size
        for i in range(batch_size):
            _, top_k_index = torch.topk(output[i], k=k)
            top_k_set = set(top_k_index.cpu().int().numpy())
            if label[i].item() in top_k_set:
                tp += 1
            else:
                fn += 1

    r = 1.0 * tp / (tp + fn)
    print("%s recall@%d: %.4f" % (data_tag, k, r))
    return r

def nDCG(output_zip_label, k, data_tag):
    return 0
