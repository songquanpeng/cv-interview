import datetime
import subprocess


def get_datetime(short=False):
    format_str = '%Y%m%d%H%M%S' if short else '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.now().strftime(format_str)


def str2bool(v):
    return v.lower() in ['true']


def get_commit_hash():
    process = subprocess.Popen(['git', 'log', '-n', '1'], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    return output[7:13]


def str2list(string, separator='-', target_type=int):
    return list(map(target_type, string.split(separator)))


def list2str(num_list):
    res = ""
    for num in num_list:
        res += f"{num:.4f},"
    if res != "":
        res = res[:-1]
    return res
