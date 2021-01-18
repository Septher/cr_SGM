import re
import json
import pandas as pd

def load_label():
    raw_label_path = "raw/labels for the laptop (specifications).xlsx"
    df = pd.read_excel(raw_label_path, sheet_name="spec", engine="openpyxl")
    raw_label = dict()
    for index, asin in enumerate(df):
        if index == 0:
            continue
        spec = df[asin]
        raw_label[asin] = {
            "screen": str(spec[0]),
            "cpu": str(spec[1]),
            "ram": str(spec[2]),
            "hdisk": str(spec[3]),
            "gcard": str(spec[4])
        }
    return raw_label

def screen_label_parser(raw_screens):
    pattern = "(\d+(?:\.\d+)?)"
    screen_dic = dict()
    for screen in raw_screens:
        size = re.findall(pattern, screen)
        screen_dic[screen] = float(size[0])
    return screen_dic

def cpu_label_parser(raw_cpus):
    freq_pattern = "(\d+(?:\.\d+)?) GHz"
    company_pattern = "(AMD|Intel)"
    gen_pattern = "Core i(\d)"
    cpu_dic = dict()
    for cpu in raw_cpus:
        freq = re.findall(freq_pattern, cpu)
        gen = re.findall(gen_pattern, cpu)
        company = re.findall(company_pattern, cpu)
        # print(cpu + "\t" + str(freq) + "\t" + str(gen) + "\t" + str(company))
        data = {}
        if len(freq) > 0:
            data["freq"] = float(freq[0])
        if len(gen) > 0:
            data["gen"] = float(gen[0])
        if len(company) > 0:
            data["company"] = company[0]
        cpu_dic[cpu] = data
    return cpu_dic

def ram_label_parser(raw_rams):
    sz_pattern = "(\d+) GB"
    gen_pattern = "DDR(\d+)"
    for ram in raw_rams:
        sz = re.findall(sz_pattern, ram)
        gen = re.findall(gen_pattern, ram)
        print(ram + "\t" + str(sz) + str(gen))

    return

def hdisk_label_smooth():
    return

def gcard_label_smooth():
    return

def label_parser(raw_label):
    parsed_label_path = "processed/labels_parsed.json"
    specs, specs_parsed = dict(), dict()
    for name in ["screen", "cpu", "ram", "hdisk", "gcard"]:
        specs[name] = set([raw_label[asin][name] for asin in raw_label])
        # print("%s: %d" % (name, len(specs[name])))

    # specs_parsed["screen"] = screen_label_parser(specs["screen"])
    # specs_parsed["cpu"] = cpu_label_parser(specs["cpu"])
    specs_parsed["ram"] = ram_label_parser(specs["ram"])
    # specs_parsed["screen"] = screen_label_parser(specs["screen"])
    # specs_parsed["screen"] = screen_label_parser(specs["screen"])
    # with open(parsed_label_path, "w") as outfile:
    #     json.dump(specs_parsed, outfile)
    #     outfile.close()


def label_smooth():
    return

def debug(items):
    pattern = "(\d+(?:\.\d+)?)"
    for item in items:
        result = re.findall(pattern, item)
        print(result)

if __name__ == '__main__':
    raw_label = load_label()
    label_parser(raw_label)
    label_smooth()