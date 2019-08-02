import csv
import nltk
import json

def cal_acc():
    gold_test = []
    with open('../data/gold_test.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        # header = next(csv_reader)
        for row in csv_reader:
            gold_test.append(row)

    xgb_test = []
    with open('../data/xgb_result.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        # header = next(csv_reader)
        for row in csv_reader:
            xgb_test.append(row)

    count = 0; correct = 0
    for item_x in xgb_test:
        for item_g in gold_test:
            if item_g[0] == item_x[0]:
                count += 1
                if item_g[1] == item_x[1] and item_g[2] == item_x[2]:
                # if common(item_g[1], item_x[1]) and item_g[2] == item_x[2]:
                    print(item_x)
                    correct += 1

    print(float(correct)/count)


def cal_acc_template():
    gold_test = []
    with open('../data/gold_test.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        header = next(csv_reader)
        for row in csv_reader:
            gold_test.append(row)

    xgb_test = []
    with open('../data/xgb_result_v1.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        header = next(csv_reader)
        for row in csv_reader:
            xgb_test.append(row)

    template_dic = {1:0, 2:0, 101:0, 151:0, 152:0}
    template_dic_right = {1:0, 2:0, 101:0, 151:0, 152:0}
    test_json = json.load( open('../data/test-data.json', 'r') )

    count = 0; correct = 0
    for item_x in xgb_test:
        for item_g in gold_test:
            if item_g[0] == item_x[0]:
                for test_item in test_json:
                    if item_g[0] == test_item['corrected_question']:
                        sparql_template_id = test_item['sparql_template_id']
                template_dic[sparql_template_id] += 1

                if item_g[1] == item_x[1] and item_g[2] == item_x[2]:
                # if common(item_g[1], item_x[1]) and item_g[2] == item_x[2]:
                    template_dic_right[sparql_template_id] += 1
                    print(item_x)

    for k, v in template_dic_right.items():
        print('the acc of template', k, float(v)/template_dic[k])


def common(a, b):
    return len(set(a.split('_')) & set(b.split('_'))) > 0


def add():
    test_json = json.load( open('../data/test-data.json', 'r') )

    xgb_test = []
    with open('../data/xgb_result.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        # header = next(csv_reader)
        for row in csv_reader:
            for item in test_json:
                if row[0] == item['corrected_question']:
                    row.append(item['sparql_query'])
                    xgb_test.append(row)

    with open('../data/new_xgb_result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["query","entity","predicate","sparql_query"])
        writer.writerows(xgb_test)



if __name__ == '__main__':

    cal_acc()
    # b = common('Homestead_Grays_Bridge', 'Homestead_Grays')
    # print(b)
    # add()
    # cal_acc_template()
