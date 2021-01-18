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

def parse_screen_label(raw_screens):
    pattern = "(\d+(?:\.\d+)?)"
    screen_dic = dict()
    for screen in raw_screens:
        size = re.findall(pattern, screen)
        screen_dic[screen] = float(size[0])
    return screen_dic

def parse_cpu_label(raw_cpus):
    freq_pattern = "(\d+(?:\.\d+)?) ?GHz"
    company_pattern = "(AMD|Intel)"
    gen_pattern = "Core i(\d)"
    cpu_dic = dict()
    for cpu in raw_cpus:
        freq = re.findall(freq_pattern, cpu)
        gen = re.findall(gen_pattern, cpu)
        # company = re.findall(company_pattern, cpu)
        data = {
            "freq": float(freq[0])
        }
        if len(gen) > 0:
            data["gen"] = float(gen[0])
        # if len(company) > 0:
        #     data["company"] = company[0]
        cpu_dic[cpu] = data
    return cpu_dic

def parse_ram_label(raw_rams):
    sz_pattern = "(\d+) ?GB?"
    gen_pattern = "DDR(\d+)"
    ram_dict = dict()
    for ram in raw_rams:
        sz = re.findall(sz_pattern, ram)
        gen = re.findall(gen_pattern, ram)
        ram_dict[ram] = int(sz[0])
    return ram_dict

def parse_hdisk_label(raw_hdisk):
    sz_pattern = "(\d+) (G|T)B?"
    hdisk_dict = dict()
    for hdisk in raw_hdisk:
        result = re.findall(sz_pattern, hdisk)
        hdisk_dict[hdisk] = int(result[0][0]) * (1024 if result[0][1] == "T" else 1)
    return hdisk_dict

def parse_gcard_label(raw_gcards):
    for gcard in raw_gcards:
        print(gcard)

    return

def parse_label(raw_label):
    parsed_label_path = "processed/labels_parsed.json"
    specs, specs_parsed = dict(), dict()
    for name in ["screen", "cpu", "ram", "hdisk", "gcard"]:
        specs[name] = set([raw_label[asin][name] for asin in raw_label])
        # print("%s: %d" % (name, len(specs[name])))

    # specs_parsed["screen"] = parse_screen_label(specs["screen"])
    # specs_parsed["cpu"] = parse_cpu_label(specs["cpu"])
    # specs_parsed["ram"] = parse_ram_label(specs["ram"])
    # specs_parsed["hdisk"] = parse_hdisk_label(specs["hdisk"])
    specs_parsed["gcard"] = parse_gcard_label(specs["gcard"])
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
    parse_label(raw_label)
    label_smooth()