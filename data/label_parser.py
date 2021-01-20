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

screen_quantile = [12, 13, 14, 16, 18]
def get_screen_id(size):
    for id, quantile in enumerate(screen_quantile):
        if size < quantile:
            return id
    return 5

def parse_screen_label(raw_screens):
    pattern = "(\d+(?:\.\d+)?)"
    screen_dic = dict()
    for screen in raw_screens:
        size = re.findall(pattern, screen)
        if len(size) != 1:
            print(screen + ": format error")
            continue
        screen_dic[screen] = get_screen_id(float(size[0]))
    print(screen_dic)
    return screen_dic

def get_cpu_id(company, freq, gen):
    if company == "AMD" or gen == "CELERON":
        if freq < 2.0:
            return 0
        if freq < 3.0:
            return 1
        return 2
    if company == "INTEL" and  (gen is not None and gen[0] == "I"):
        if gen[1] == "3":
            return 3 if freq < 2.4 else 4
        if gen[1] == "5":
            if freq <= 2.0:
                return 5
            return 6 if freq < 3.0 else 7
        if gen[1] == "7":
            if freq <= 2.0:
                return 6
            return 7 if freq < 3.0 else 8
    return 9

def parse_cpu_label(raw_cpus):
    freq_pattern = "(\d+(?:\.\d+)?) ?GHZ"
    company_pattern = "(AMD|INTEL)"
    gen_pattern = "I\d|CELERON"
    cpu_dic = dict()
    for cpu in raw_cpus:
        freq = re.findall(freq_pattern, cpu.upper())
        gen = re.findall(gen_pattern, cpu.upper())
        company = re.findall(company_pattern, cpu.upper())
        if len(freq) != 1:
            print(cpu + ": format error" + str(freq))
            continue
        cpu_dic[cpu] = get_cpu_id(company[0] if len(company) == 1 else None, float(freq[0]), gen[0] if len(gen) == 1 else None)
    return cpu_dic

def get_ram_id(size, gen):
    if size <= 4:
        return 0
    if size <= 6:
        return 1
    if size <= 8:
        if gen is not None and gen == 4:
            return 3
        else:
            return 2
    if size <= 12:
        return 4
    return 5

def parse_ram_label(raw_rams):
    sz_pattern = "(\d+) ?GB?"
    gen_pattern = "DDR(\d+)"
    ram_dict = dict()
    for ram in raw_rams:
        sz = re.findall(sz_pattern, ram)
        gen = re.findall(gen_pattern, ram)
        ram_dict[ram] = get_ram_id(int(sz[0]), int(gen[0]) if len(gen) == 1 else None)
    print(ram_dict)
    return ram_dict

SSD_quantile = [16, 32, 128, 256, 512]
HDD_quantile = [256, 512, 1024]
def get_hdisk_id(size, tp):
    if tp == "SSD": #[0, 5]
        for id, quantile in enumerate(SSD_quantile):
            if size <= quantile:
                return id
        return 5
    else: #[6, 9]
        for id, quantile in enumerate(HDD_quantile, 6):
            if size <= quantile:
                return id
        return 9

def parse_hdisk_label(raw_hdisk):
    sz_pattern = "(\d+) (G|T)B?"
    hdisk_dict = dict()
    for hdisk in raw_hdisk:
        s = hdisk.upper()
        tp = "SSD" if "SOLID" in s or "SSD" in s or "EMMC" in s else "HDD"
        result = re.findall(sz_pattern, s)
        if len(result) != 1:
            print(hdisk + ": format error")
            continue
        size = int(result[0][0]) * (1024 if result[0][1] == "T" else 1)
        hdisk_dict[hdisk] = get_hdisk_id(size, tp)
        print(hdisk + "\t" + str(size) + "\t" + tp + "\t" + str(hdisk_dict[hdisk]))
    return hdisk_dict

def get_gcard_id(company, series, num):
    if company == "NVIDIA" or series == "GTX":
        return 0 if num is not None and int(num) >= 1000 else 1
    if company == "AMD" or series == "RADEON":
        return 2 if num is not None and int(num) <= 4 else 3
    if company == "INTEL" or series == "INTEGRATED":
        if num is None or series == "INTEGRATED":
            return 4
        return 5 if int(num[0]) <= 5 else 6
    return 7

def parse_gcard_label(raw_gcards):
    company_pattern = "(INTEL|AMD|NVIDIA)"
    series_pattern = "INTEGRATED|GTX|RADEON"
    num_pattern = "(R|GTX|GRAPHICS|GEFORCE) ?(\d+)"
    gcard_dict = dict()
    for gcard in raw_gcards:
        company = re.findall(company_pattern, gcard.upper())
        series = re.findall(series_pattern, gcard.upper())
        nums = re.findall(num_pattern, gcard.upper())
        gcard_dict[gcard] = get_gcard_id(company[0] if len(company) == 1 else None, series[0] if len(series) == 1 else None, nums[0][1] if len(nums) == 1 else None)
    return gcard_dict

def parse_label(raw_label):
    parsed_label_path = "processed/labels_parsed.json"
    specs, specs_parsed = dict(), dict()
    for name in ["screen", "cpu", "ram", "hdisk", "gcard"]:
        specs[name] = set([raw_label[asin][name] for asin in raw_label])
        # print("%s: %d" % (name, len(specs[name])))

    # specs_parsed["screen"] = parse_screen_label(specs["screen"])
    specs_parsed["cpu"] = parse_cpu_label(specs["cpu"])
    # specs_parsed["ram"] = parse_ram_label(specs["ram"])
    # specs_parsed["hdisk"] = parse_hdisk_label(specs["hdisk"])
    # specs_parsed["gcard"] = parse_gcard_label(specs["gcard"])
    # with open(parsed_label_path, "w") as outfile:
    #     json.dump(specs_parsed, outfile)
    #     outfile.close()


def debug(items):
    pattern = "(\d+(?:\.\d+)?)"
    for item in items:
        result = re.findall(pattern, item)
        print(result)

if __name__ == '__main__':
    raw_label = load_label()
    parse_label(raw_label)
