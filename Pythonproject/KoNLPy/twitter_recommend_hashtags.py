	#Author-Abhishek Roy
	#Computer science and department
	#Texas A & M 
import tweepy
import time
# from urllib2 import urlopen
import urllib.request
from bs4 import BeautifulSoup
import json
from collections import defaultdict
import operator
	
# Extracting tweets and timestamp by crawling on the user's twitter page
def extractTweetsByCrawlingPage(userName):
	print ("===============================")
	countTime = 0
	countTweet = 0
	url = 'https://twitter.com/' + userName
	f = urllib.request.Request(url)	
	soup = BeautifulSoup(f.read())
	f.close()
	tweets = soup.findAll('p', {'class': 'ProfileTweet-text js-tweet-text u-dir'})
	timeStamps = soup.findAll('a', {'class': 'ProfileTweet-timestamp js-permalink js-nav js-tooltip'})
	for a in timeStamps:
		print (a.get('title'), "\n")
		countTime = countTime + 1
	
	for x in tweets:
		if x.findAll('b'):
			countTweet = countTweet + 1
			for hash in x.findAll('b'):
				print (hash.renderContents())
		else:
			print ("none")
		
		print ("\n")
	
	print ("countTime: ", countTime)
	print ("countTweet: ", countTweet)
	print ("===============================")
	
# Authenticating twitter API
def Authenticate_twitter():    
    #Authenticating step
    consumer_key = "T7HwaOviiMC2PznkqqpsKWsel "
    consumer_secret = "rgoJu17wAvV0y66XzLTwjCqcW9xJMXT0wEmvnUSpRVMpiE13Xu"
    access_key = "121943817-jCc1ABDeQ08sUWWxzBrgYeALDhCqNSUVSSxN6OV0"
    access_secret = "dtiROTXDjNxfcJmERS0XwtXalC8xeUyCQ2whRcJpo2tBI"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return tweepy.API(auth)

# Updating score - CumulativeCount for each hashtag
def CumulativeCount(tag, key, seed, G):	
	return 1
	
# Updating score - IntersectingGroupCount for each hashtag
def InstersectingGroupCount(tag, key, seed, G):
	for tag in G.get(key):		# each tag corresponds to each hashtag
		if tag in seed:
			return 1
	return 0
    		
def main():
    
    while 1==1:
        userName = raw_input("Enter Twitter username (0 to EXIT): ")
        if userName == '0':
        	break
        	
        initHash = raw_input("Enter base Hashtag: ") # initial Hashtag for which recommendation needed
        seed = []							# initializing a list
        seed.append(initHash.lower())      	# converting input hashtag to lower case
        
        # Extracting tweets and timestamp by crawling on the user's twitter page
        # extractTweetsByCrawlingPage(userName)
         	
    	# Extracting tweets and timestamp by using twitter API
    	api=Authenticate_twitter()
    	G = defaultdict(list)	# collections of tweets with each having hashtags
    	TT = defaultdict(list)	# tweet-time or tweet created at
    	
    	it = 0					# counter for number of pulled tweets
    	id = 0					# counter for number of tweets with hashtags
    	for tweet in api.user_timeline(screen_name=userName, count=5000, include_rts="false"):
    		if it < 5000 :
    			if [hashtag.get('text') for hashtag in tweet.entities["hashtags"]]:
    				# print tweet.created_at
    				TT[id] = tweet.created_at
    				# print tweet.text
    				for hashtag in tweet.entities["hashtags"]:
    					# print hashtag.get('text')
    					G[id].append(str(hashtag.get('text').lower()))
    				
    				id = id + 1
    			
    		else:
    			break
    		
    		it = it + 1
    		# print "\n"
    	
    	# Method selection for Updating score
    	print ("Default method for updating score: IntersectingGroupCount")
        scoringMethod = raw_input("switch to CumulativeCount? (y/n): ")
    	
    	# Core Routine - Main algorithm
    	isNextHashInterested = 'y'
    	while isNextHashInterested == 'y':
    		F = defaultdict(list)	# dictionary with hashtags as keys and scores as values
    		for key, value in G.iteritems():		# each key corresponds to each tweet
    			for tag in G.get(key):				# each tag corresponds to each hashtag
    				if tag not in seed:
    					if F.get(tag, 'none') == 'none' :
    						F[tag] = 0
    					if scoringMethod == 'y' :
    						F[tag] = F[tag] + CumulativeCount(tag, key, seed, G)
    					else:
    						F[tag] = F[tag] + InstersectingGroupCount(tag, key, seed, G)
    	
    		# Hashtag Recommendation
    		print ("Recommended hashtag: ", max(F.iteritems(), key=operator.itemgetter(1))[0]) 
    	
    		# Option to have more hashtag recommendations
    		isNextHashInterested = raw_input("Interested in next Hashtag Recommendation? (y/n): ")
    		if isNextHashInterested == 'y' :
	    		isHashUsed = raw_input("Used the recommended hashtag? (y/n): ")
    			if isHashUsed == 'y' :
    				seed.append(max(F.iteritems(), key=operator.itemgetter(1))[0])
    			else:
    				hashUsed = raw_input("Enter the hashtag used: ")
    				seed.append(hashUsed)

main()