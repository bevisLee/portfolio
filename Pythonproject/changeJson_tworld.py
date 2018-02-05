import json
import operator
import csv

def main():
    lastData = 0
    dataIndex = {}
    data = transData('merge_new.json')

    data = deleteObj(data)

    for i in list(data.keys()):
        if lastData == 0:
            lastData = []
            lastData.append(data[i])
            dataIndex[0] = 1

        else:
            count = 0
            flag = False
            for x in lastData:
                if data[i] == x:
                    dataIndex[count] = dataIndex[count] + 1
                    flag = True
                    break
                count = count+1
            if flag == False:
                lastData.append(data[i])
                dataIndex[count] = 1

#    print(dataIndex)
#    print(sorted(dataIndex.items(), key=operator.itemgetter(1)))

    max = len(sorted(dataIndex.items(), key=operator.itemgetter(1)))-1

    #while max > len(sorted(dataIndex.items(), key=operator.itemgetter(1)))-100:
    #    print(sorted(dataIndex.items(), key=operator.itemgetter(1))[max][1])
    #    print(lastData[sorted(dataIndex.items(), key=operator.itemgetter(1))[max][0]])
    #    max = max-1

    with open('pathData_without_query.csv', 'w') as csvfile:
        fieldnames = ['path', 'num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        i = 0
        while max >= 0:
            num = sorted(dataIndex.items(), key=operator.itemgetter(1))[max][1]
            fullPath = ""
            for idx, data in enumerate(lastData[sorted(dataIndex.items(), key=operator.itemgetter(1))[max][0]]):
                if idx == len(lastData[sorted(dataIndex.items(), key=operator.itemgetter(1))[max][0]])-1:
                    fullPath += data
                else:
                    fullPath += data + " > "

            writer.writerow({'path':fullPath, 'num':num})

            i += 1

            max -= 1





def deleteObj(data):
    for i in list(data.keys()):
        flag = False
        for x in data[i]:
            if 'www.tworld.co.kr/normal.do?serviceId=S_PROD2001&viewId=V_PROD2001&prod_id=NA' in x:
                flag = True
        if flag == False:
            del data[i]

    return data

def transData(dataDirectory):
    visitId = 0
    idx = 0
    flag = False
    data = {}
    with open(dataDirectory, encoding="ISO-8859-1") as f:
        for line in f:
            if visitId == json.loads(line)['visitId']:
                if flag == False:
                    if 'www.tworld.co.kr/normal.do?serviceId=S_PROD2001&viewId=V_PROD2001&prod_id=NA' in json.loads(line)['hits_page_pagePath']:
                        #data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
                        data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
                        idx += 1
                        flag = True
                    else:
                        data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
            else:
                flag = False
                visitId = json.loads(line)['visitId']
                data[str(idx)] = []
                if 'www.tworld.co.kr/normal.do?serviceId=S_PROD2001&viewId=V_PROD2001&prod_id=NA' in json.loads(line)['hits_page_pagePath']:
                    #data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
                    data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
                    idx += 1
                    flag = True
                else:
                    data[str(idx)].append(json.loads(line)['hits_page_pagePath'])
    return data


if __name__ == "__main__":
    main()