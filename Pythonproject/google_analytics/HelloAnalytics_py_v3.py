
# web page - http://www.marinamele.com/use-google-analytics-api-with-python

import httplib2
from apiclient.discovery import build
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client import tools
import argparse
 
CLIENT_SECRETS = 'C:/Users/bevis/Downloads/client_secret_72227845996-8o20ovjnnftc2vrv35vv75f1jjin1cn9.apps.googleusercontent.com.json'
viewid = "159480009" 

# The Flow object to be used if we need to authenticate.
FLOW = flow_from_clientsecrets(
    CLIENT_SECRETS,
    scope='https://www.googleapis.com/auth/analytics.readonly',
    message='%s is missing' % CLIENT_SECRETS
    )
 
# A file to store the access token
TOKEN_FILE_NAME = 'credentials.dat'
 
 
def prepare_credentials():
    parser = argparse.ArgumentParser(parents=[tools.argparser])
    flags = parser.parse_args()
    # Retrieve existing credendials
    storage = Storage(TOKEN_FILE_NAME)
    credentials = storage.get()
    # If no credentials exist, we create new ones
    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(FLOW, storage, flags)
    return credentials
 
 
def initialize_service():
    # Creates an http object and authorize it using
    # the function prepare_creadentials()
    http = httplib2.Http()
    credentials = prepare_credentials()
    http = credentials.authorize(http)
    # Build the Analytics Service Object with the authorized http object
    return build('analytics', 'v3', http=http)
 
if __name__ == '__main__':
    service = initialize_service()

## session 
def get_sessions(service, profile_id, start_date, end_date):
    ids = "ga:" + profile_id
    metrics = "ga:sessions"
    data = service.data().ga().get(
        ids=ids, start_date=start_date, end_date=end_date, metrics=metrics
        ).execute()
    return data["totalsForAllResults"][metrics]
 
if __name__ == '__main__':
    service = initialize_service()
    profile_id = viewid
    print (get_sessions(service, profile_id, "2017-12-01", "2017-12-31"))

## traffic source
def get_source_group(service, profile_id, start_date, end_date):
    ids = "ga:" + profile_id
    metrics = "ga:sessions"
    dimensions = "ga:channelGrouping"
    data = service.data().ga().get(
        ids=ids, start_date=start_date, end_date=end_date, metrics=metrics,
        dimensions=dimensions).execute()
    return dict(data["rows"] + [["total", data["totalsForAllResults"][metrics]]])
 
 
if __name__ == '__main__':
    service = initialize_service()
    profile_id = viewid
    start_date = "2017-12-01"
    end_date = "2017-12-31"
    data = get_source_group(service, profile_id, start_date, end_date)
    for key, value in data.iteritems():
        print (key, value)

## traffic source2
not_source_filters = {
    "social": "ga:hasSocialSourceReferral==No",
    "organic": "ga:medium!=organic",
    "direct":  "ga:source!=(direct),ga:medium!=(none);ga:medium!=(not set)",
    "email": "ga:medium!=email",
    "referral": "ga:medium!=referral,ga:hasSocialSourceReferral!=No"
}
 
source_filters = {
    "social": "ga:hasSocialSourceReferral==Yes",
    "organic": "ga:medium==organic",
    "direct":  "ga:source==(direct);ga:medium==(none),ga:medium==(not set)",
    "email": "ga:medium==email",
    "referral": "ga:medium==referral;ga:hasSocialSourceReferral==No",
    "other": "%s;%s;%s;%s;%s" % (
        not_source_filters["social"], not_source_filters["organic"],
        not_source_filters["direct"], not_source_filters["email"],
        not_source_filters["referral"])
}
 
 
def get_source_sessions(service, profile_id, start_date, end_date, source):
    ids = "ga:" + profile_id
    metrics = "ga:sessions"
    filters = source_filters[source]
    data = service.data().ga().get(
        ids=ids, start_date=start_date, end_date=end_date, metrics=metrics,
        filters=filters).execute()
    return data["totalsForAllResults"][metrics]
 
 
if __name__ == '__main__':
    service = initialize_service()
    profile_id = viewid
    start_date = "2017-12-01"
    end_date = "2017-12-31"
    for source in ["social", "organic", "direct", "email", "referral", "other"]:
        print (source, get_source_sessions(
            service, profile_id, start_date, end_date, source))