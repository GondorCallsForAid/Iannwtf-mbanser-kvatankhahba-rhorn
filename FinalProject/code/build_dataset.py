# Libraries and Settings
import os
import re
import pytube # python3 -m pip install --upgrade "git+https://github.com/nficano/pytube.git"
from pytube import YouTube 
import urllib3
http = urllib3.PoolManager()


# Categories from which videos should be fetched
# more explicit categories tend to be better
# alternative queries: "games with ambient sound"
categories = ["black forest", "ocean", "deers", "birds", "train", "city traffic", "diving", "waves"]

# Create folders for videos
path = f"videos/"
for c in categories:
	try:
	    os.makedirs(f"{path}{c}")
	except FileExistsError:
		pass
	except OSError:
	    print (f"Creation of the directory {path}{c} failed")
	else:
	    print (f"Successfully created the directory {path}{c}")

# Query for videos (~30 videos)
videos = {} # {category: video_url}
for c in categories:
	qc = "+".join(c.split(" "))
	query = f"intitle%3\”{qc}+raw+footage\”"
	print(f"Downloading {c} videos based on query: {query}")
	html = http.request("GET", "https://www.youtube.com/results?search_query=" + query)
	videos[c] = [f"https://www.youtube.com/watch?v={video_id}" for video_id in re.findall(r"watch\?v=(\S{11})", html.data.decode())]

	# filter videos
	max_length = 1800 #sec
	max_views = 1000000
	i = 0
	for url in videos[c]:
		try:
			video = YouTube(url)
			filter_words = ["music", "drone", "lyric"] # copyright
			if video.length > max_length or video.views > max_views or any([w in video.description.lower() for w in filter_words]):
				videos[c].remove(url)
			else:
				# select appropriate stream
				stream = video.streams.filter(abr="96kbps")[0] # video is fine for 96kps as far i can tell (360p)
				# download video into according folder video_index_ytid
				ytid = url.split("=")[1]
				stream.download(output_path=f"{path}{c}", filename = f"{c}_video_{i}_{ytid}")
				i += 1
		except pytube.exceptions.VideoUnavailable:
			print(f"{url} not available")
		

print("\nAll downloads finished")
