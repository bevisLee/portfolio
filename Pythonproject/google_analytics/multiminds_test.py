
# web page - http://www.multiminds.eu/2016/08/04/tutorial-google-analytics-reporting-api-python/

from oauth2client.service_account import ServiceAccountCredentials
from apiclient.discovery import build
 
import httplib2
import matplotlib.pyplot as plt
import numpy as np

#create service credentials
#this is where you'll need your json key
#replace "keys/key.json" with the path to your own json key
# iam - https://console.developers.google.com/iam-admin/serviceaccounts/project?project=bevis-sandbox
credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/bevis/Downloads/', ['https://www.googleapis.com/auth/analytics.readonly'])
 
# create a service object you'll later on to create reports
http = credentials.authorize(httplib2.Http())
service = build('analytics', 'v4', http=http, discoveryServiceUrl=('https://analyticsreporting.googleapis.com/$discovery/rest'))

response = service.reports().batchGet(
    body={
        'reportRequests': [
            {
                'viewId': '159480009',
                'dateRanges': [{'startDate': '2017-12-01', 'endDate': '2017-12-10'}],
                'metrics': [{'expression': 'ga:sessions'}],
                'dimensions': [{"name": "ga:city"}],
                'orderBys': [{"fieldName": "ga:sessions", "sortOrder": "DESCENDING"}],
                'pageSize': 10
            }]
    }
).execute()

#create two empty lists that will hold our city and visits data
cities = []
val = []
 
#read the response and extract the data we need
for report in response.get('reports', []):
 
    columnHeader = report.get('columnHeader', {})
    dimensionHeaders = columnHeader.get('dimensions', [])
    metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])
    rows = report.get('data', {}).get('rows', [])
 
    for row in rows:
 
        dimensions = row.get('dimensions', [])
        dateRangeValues = row.get('metrics', [])
 
        for header, dimension in zip(dimensionHeaders, dimensions):
            cities.append(dimension)
 
        for i, values in enumerate(dateRangeValues):
            for metricHeader, value in zip(metricHeaders, values.get('values')):
                val.append(int(value))

#reverse the order of the data to create a nicer looking graph
val.reverse()
cities.reverse()
 
#create a horizontal bar chart
plt.barh(np.arange(len(cities)), val, align='center', alpha=0.4)
plt.yticks(np.arange(len(cities)), cities)
 
#add some context
plt.xlabel('Visits')
plt.title('Top 10 cities last 30 days')
 
#render the damn thing!
plt.show()