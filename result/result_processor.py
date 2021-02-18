import json
from model.params import DEVICE_ORDER

file2description = {
    "02-19-10-24.json": "baseline",
    "02-19-10-32.json": "baseline",
    "02-19-10-39.json": "baseline"
    # ,
    # "02-19-11-03.json": "change_device_order",
    # "02-19-11-10.json": "change_device_order",
    # "02-19-11-20.json": "change_device_order"
}

def process():
    title = "embedding_size_512"
    files = ["02-20-12-47", "02-20-13-03", "02-20-13-09"]
    output = {"overall": {}}
    for device in DEVICE_ORDER:
        output[device] = {}
    for file in files:
        with open("test_result/%s.json" % file, "r") as fd:
            d = json.load(fd)
            for device in DEVICE_ORDER:
                data = d[device]
                for i in range(1, 6):
                    for c in ["recall", "precision", "nDCG"]:
                        criterion = "%s@%d" % (c, i)
                        if criterion not in output[device]:
                            output[device][criterion] = 0.0
                        if criterion not in output["overall"]:
                            output["overall"][criterion] = 0.0
                        v = float("%.2f" % (data[criterion] * 100.0 / len(files)))
                        output[device][criterion] += v
                        output["overall"][criterion] += float("%.2f" % (v / 5))
                fd.close()
    with open("processed_test_result/%s.json" % title, "w") as outfile:
        json.dump(output, outfile)
        outfile.close()
process()