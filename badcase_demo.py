import urllib.parse
import urllib.request
import numpy as np
from sklearn import metrics
import json

# media = ['audio', 'video', 'fm']
categories = ['alerts', 'all_control', 'baike', 'calculator', 'call', 'chat', 'cook_book', 'audio', 'video', 'fm',
              'news', 'road',
              'shopping', 'calendar', 'translator', 'weather', 'car_limit', 'train', 'travel', 'stock', 'third_party']
class_num = 21


def generate(str):
    str = str.replace('\n', '')
    # if str in media:
    # str = 'media'
    result = categories.index(str)
    return result


base_url = 'http://jnlu-core-proxy.jd.com/alpha_broker/handle'

y_true = np.zeros(shape=100000, dtype=np.int32)
y_pred = np.zeros(shape=100000, dtype=np.int32)

data_path = 'music_broadcast_1.txt'

count = 0
with open(data_path,'r',encoding='utf-8') as f:
    with open('badcase_yinpin', 'w') as fw:
        for line in f:
            # count+=1
            # if(count<1301):
            #    continue
            data = line.strip().split("\t")
            if data.__len__() != 2:
                print(line)
                continue
            test_case = data[0]
            pred = data[1]
            values = {'env': 'yfb2',
                      'state': 'INIT_STATE',
                      'text': test_case}
            data = urllib.parse.urlencode(values)
            data = data.encode('ascii')  # data should be bytes
            req = urllib.request.Request(base_url, data)
            with urllib.request.urlopen(req) as response:
                result = response.read().decode("utf-8")
                result = json.loads(result)
                top_domain = result['domainList']
                real_domain = top_domain[0]['domainName']
                y_pred[count] = generate(real_domain)
                y_true[count] = generate(pred)
                count += 1
                pred1 = pred.strip()
                if real_domain != pred1:
                    fw.write(test_case + '\t' + real_domain + '\t' + pred1 + '\n')
                #print('error:', test_case)
                print(count)

print("Precision, Recall and F1-Score...")
print(metrics.classification_report(y_true, y_pred, target_names=categories, digits=4))