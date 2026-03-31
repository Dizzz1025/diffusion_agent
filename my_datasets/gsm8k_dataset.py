import re

def gsm_data_process(dataset):
    # extract the question, step and answer
    list_data_dict = []
    for data in dataset:
        item = {"task":data["question"]}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)

    return list_data_dict


def gsm_get_predict(pred_str):
    def extract_boxed_content(s):
        ans = s.split('boxed')[-1].strip()
        if not ans:
            return ''
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a
        return ans

    pred = ''

    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        pred = extract_boxed_content(pred_str)
    else:
        # 先整体匹配带 $ / 逗号 / 小数 / 负号 的数字
        matches = re.findall(r'-?\$?\d[\d,]*\.?\d*', pred_str)
        pred = matches[-1] if matches else ''

    pred = _strip_string(pred)
    pred = pred.replace(',', '').replace('$', '').strip()

    if pred.endswith('.'):
        pred = pred[:-1]
    if pred.endswith('/'):
        pred = pred[:-1]

    # 再兜底处理一次 boxed
    if 'boxed' in pred:
        pred = extract_boxed_content(pred)
        pred = _strip_string(pred)
        pred = pred.replace(',', '').replace('$', '').strip()

    # 最终提取一个合法数字：支持负数和小数
    matches = re.findall(r'-?\d+\.?\d*', pred)
    return matches[-1] if matches else '0'


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string