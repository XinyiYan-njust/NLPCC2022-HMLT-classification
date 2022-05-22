import json

print("打开字典...")
with open(r'label_index/level1_dict.json', 'r', encoding='utf8') as fp:
    level1_dict = json.load(fp)
level1_keys = list(level1_dict.keys())
level1_values = list(level1_dict.values())

with open(r'label_index/level2_dict.json', 'r', encoding='utf8') as fp:
    level2_dict = json.load(fp)
level2_keys = list(level2_dict.keys())
level2_values = list(level2_dict.values())

with open(r'label_index/level3_dict.json', 'r', encoding='utf8') as fp:
    level3_dict = json.load(fp)
level3_keys = list(level3_dict.keys())
level3_values = list(level3_dict.values())

with open(r'verification set.json', 'r', encoding='utf8') as fp:
    test_data = json.load(fp)

sample_list = []
with open(r'../Test_sample.json', 'r', encoding='utf8') as fp:
    # Test_sample = json.load(fp)
    count = 0
    while True:
        if count % 100 == 0:
            print(count)
        count += 1
        line = fp.readline()
        if not line:
            break
        sample_list.append(json.loads(line))

pred_list = []
with open(r'../../HARNN/output/1652786880/predictions0.24.json', 'r', encoding='utf8') as fp:
    # predictions = json.load(fp)
    count = 0
    while True:
        if count % 100 == 0:
            print(count)
        count += 1
        line = fp.readline()
        if not line:
            break
        pred_list.append(json.loads(line))
        # u_id = pred_dict['id']
        # pred_dict['labels'].remove(1804)
        # pred_dict['labels'].append(21 + 298 + 1475 + 1)
        # with open(r'Validation_sample2.json', 'a', encoding='utf-8') as fp2:
        #     json.dump(pred_dict, fp2, ensure_ascii=False)
        #     fp2.write('\n')

print(f"pred_list: {len(pred_list)}")
print(f"sample_list:{len(sample_list)}")
print(f"test_data:{len(test_data)}")

result_list = []
for index, temp_dict in enumerate(test_data):
    # u_id1 = pred_list[index]['id']
    # u_id2 = sample_list[index]['id']
    # if u_id1 != u_id2:
    #     print(f"u_id1:{u_id1}\nu_id2:{u_id2}")
    #     input("请检查。。。。")
    # title1 = sample_list[index]['title']
    # title2 = temp_dict['title']
    # if title1 != title2:
    #     print(f"title1{title1}\ntitle2{title2}")
    #     input("请检查。。。")

    pred_index_list = pred_list[index]['predict_labels']
    pred_label_list = []
    for label_index in pred_index_list:
        if label_index <= 21:
            pred_label_list.append(level1_keys[level1_values.index(label_index)])
        elif label_index <= 319:
            pred_label_list.append(level2_keys[level2_values.index(label_index-21)])
        elif label_index <= 1794:
            pred_label_list.append(level3_keys[level3_values.index(label_index-21-298)])
    # if len(pred_label_list) != len(pred_index_list):
    #     input("预测标签与索引列表长度不同！")

    temp_dict['pred_labels'] = pred_label_list
    result_list.append(temp_dict)

print(f"result_list:{len(result_list)}")
with open(r'../evaluation code/input_sample/1652786880/results0.24.json', 'w', encoding='utf8') as fp:
    json.dump(result_list, fp, ensure_ascii=False)
